# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# GDN in_proj_ba (BF16 [96,2560], cuBLAS wmma + splitK reduce) -> tuned Triton BF16 GEMV for M<=16. Same weights, fp32 accum.
# /exp/ba.txt: on | off | tune
import importlib.util, torch, gc, itertools
s_ = importlib.util.spec_from_file_location("exp_bf16gemv", "/exp/exp_bf16gemv.py"); BG = importlib.util.module_from_spec(s_); s_.loader.exec_module(BG)
r = worker.model_runner; model = r.model
mode = open("/exp/ba.txt").read().split()[0]
mods = [l.linear_attn.in_proj_ba for l in model.language_model.model.layers if hasattr(l, "linear_attn")]
def timeit(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): fn()
    g.replay(); torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / n * 1000
out = []
x = torch.randn(16, 2560, dtype=torch.bfloat16, device="cuda")
m0 = mods[0]; w0 = m0.weight
cfg = getattr(model, "_exp_ba_cfg", None)
if cfg is None or mode == "tune":
    for M in (4, 1):
        xx = x[:M]
        # cycle over all 36 weights so it is not L2-hot
        base = timeit(lambda: [torch.nn.functional.linear(xx, m.weight) for m in mods]) / len(mods)
        best = None
        for BN, BK, KS, nw in itertools.product((8, 16, 32), (128, 256, 512), (1, 2, 4), (2, 4)):
            if (2560 // KS) % BK: continue   # split-K slice must be a multiple of BK (else OOB reads -> garbage)
            try:
                _g = BG.linear(xx, w0, (BN, BK, KS, nw)).float(); _r = torch.nn.functional.linear(xx.float(), w0.float())
                if not ((_g - _r).norm() / _r.norm()).item() < 1e-2: continue
                t = timeit(lambda: [BG.linear(xx, m.weight, (BN, BK, KS, nw)) for m in mods]) / len(mods)
            except Exception:
                continue
            if best is None or t < best[0]: best = (t, (BN, BK, KS, nw))
        ref = torch.nn.functional.linear(xx.float(), w0.float()); got = BG.linear(xx, w0, best[1]).float()
        out.append(f"M={M}: F.linear {base:.1f}us, triton best {best[0]:.1f}us cfg {best[1]} relerr {((got-ref).norm()/ref.norm()).item():.1e}")
        if M == 4: cfg = best[1]
    model._exp_ba_cfg = cfg
if mode == "on":
    for m in mods:
        w = m.weight
        def fwd(x, _w=w, _o=type(m).forward.__get__(m)):
            if x.reshape(-1, x.shape[-1]).shape[0] > 16: return _o(x)
            return BG.linear(x, _w, cfg), None
        m.forward = fwd
elif mode == "off":
    for m in mods: m.__dict__.pop("forward", None)
if mode in ("on", "off"):
    for mg in [r.cudagraph_manager, r.speculator.prefill_cudagraph_manager, r.speculator.decode_cudagraph_manager]: mg.graphs.clear()
    gc.collect(); r.capture_model(); out.append(f"ba {mode} ({len(mods)} modules), cfg {cfg}, recaptured")
RESULT = "\n".join(out)
