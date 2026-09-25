# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# Draft head coarse INT2 g16 + FP32 refine from FP8 rows (top-K). /exp/draft2.txt: "int2r K [verify]" | "int4r" (restores E32 fn)
import importlib.util, torch, sys, gc, itertools
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
h2 = load("exp_head2", "/exp/exp_head2.py")
r = worker.model_runner; m = r.speculator.model
mod = sys.modules[type(m).__module__]
arg = open("/exp/draft2.txt").read().split()
out = []
if not hasattr(mod, "_exp_int4r_fn"): mod._exp_int4r_fn = mod._exp_fp8_logits   # E32 installed fn
orig_fp8 = mod._exp_fp8_logits_orig
if not hasattr(m, "_exp_draft_int2"):
    tl_ = getattr(r.model, "lm_head", None) or r.model.language_model.lm_head
    w = tl_.weight.index_select(0, m._draft_id_to_target_id.long())
    m._exp_draft_int2 = h2.quantize(w); del w
P, S = m._exp_draft_int2; N = P.shape[0]
def timeit(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): fn()
    g.replay(); torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / n * 1000
x1 = torch.randn(1, 2560, dtype=torch.bfloat16, device="cuda")
if not hasattr(m, "_exp_d2cfg"):
    best = None
    for BN, BK, nw in itertools.product((16, 32, 64), (256, 512), (2, 4)):
        try: t = timeit(lambda: h2.logits(x1, P, S, N, BN, BK, nw))
        except Exception: continue
        if best is None or t < best[0]: best = (t, BN, BK, nw)
    m._exp_d2cfg = best[1:]; out.append(f"int2 draft coarse M=1 {best[0]:.0f}us cfg {best[1:]}")
cfg = m._exp_d2cfg
if not hasattr(m, "_exp_d2cnt"): m._exp_d2cnt = torch.zeros(3, dtype=torch.long, device="cuda")
cnt = m._exp_d2cnt
if arg[0] == "int2r":
    K = int(arg[1]); VER = "verify" in arg
    def f(x, fp8, scale):
        x = x.to(torch.bfloat16).contiguous()
        ap = h2.logits(x, P, S, N, *cfg)
        idx = ap.topk(K, dim=1).indices; mm = x.shape[0]
        w = fp8.index_select(0, idx.reshape(-1)).view(mm, K, -1).to(torch.float32)
        ex = torch.bmm(w, x.float().unsqueeze(-1)).squeeze(-1) * scale[idx]
        o = torch.full_like(ap, -1e30); o.scatter_(1, idx, ex)
        if VER:
            ref = orig_fp8(x, fp8, scale); got = o.argmax(1)
            gap = ref.max(1).values - ref.gather(1, got[:, None]).squeeze(1)
            cnt[0] += mm; cnt[1] += (ref.argmax(1) != got).sum(); cnt[2] += (gap > 1e-3).sum()
        return o
    mod._exp_fp8_logits = f
else:
    mod._exp_fp8_logits = mod._exp_int4r_fn
out.append(f"mode {arg}: head M=1 {timeit(lambda: mod._exp_fp8_logits(x1, m._draft_lm_head_fp8, m._draft_lm_head_fp8_scale)):.0f}us")
for mg in [r.cudagraph_manager, r.speculator.prefill_cudagraph_manager, r.speculator.decode_cudagraph_manager]: mg.graphs.clear()
gc.collect(); r.capture_model()
out.append(f"cnt(rows, flips, flips gap>1e-3)={cnt.tolist()}")
RESULT = "\n".join(out)
