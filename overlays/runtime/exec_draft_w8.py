# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# Draft-only: swap BF16 dense Linears in the MTP draft model (fc_*, self_attn qkv/o/index, shared expert, router gate) to W8A16 g64.
# Output-exact by construction (greedy verification); may change acceptance. Mode /exp/draft_w8.txt: install | status
import importlib.util, torch, gc, time, itertools
spec = importlib.util.spec_from_file_location("exp_hc_w8", "/exp/exp_hc_w8.py")
W8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(W8)
r = worker.model_runner; dm = r.speculator.model
mode = open("/exp/draft_w8.txt").read().split()[0]
names = ["model.fc_embedding", "model.fc_hidden", "model.layers.0.self_attn.qkv_proj", "model.layers.0.self_attn.o_proj",
         "model.layers.0.self_attn.indexer.index_qk_proj", "model.layers.0.mlp.shared_expert.gate_up_proj",
         "model.layers.0.mlp.shared_expert.down_proj", "model.layers.0.mlp.gate"]
def timeit(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10): fn()
    g.replay(); torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / 10 * 1000
out = []
if mode == "install":
    tot_b = tot_w = 0
    for n in names:
        mod = dm.get_submodule(n)
        if getattr(mod, "_exp_w8", None) is not None: continue
        w = mod.weight.data; N, K = w.shape
        x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")
        q, s = W8.quantize(w)
        tb = timeit(lambda: torch.nn.functional.linear(x, w))
        best = None
        for BN, BK, KS, nw in itertools.product((16, 32, 64), (64, 128, 256), (1, 2, 4), (2, 4)):
            if K % (BK * KS): continue
            try: t = timeit(lambda: W8.linear(x, q, s, (BN, BK, KS, nw)))
            except Exception: continue
            if best is None or t < best[0]: best = (t, (BN, BK, KS, nw))
        ref = torch.nn.functional.linear(x, w).float(); y = W8.linear(x, q, s, best[1]).float()
        err = ((y - ref).norm() / ref.norm()).item()
        cfg = best[1]
        def fwd(xx, _q=q, _s=s, _cfg=cfg, _mod=mod):
            y = W8.linear(xx.reshape(-1, xx.shape[-1]), _q, _s, _cfg).view(*xx.shape[:-1], -1)
            b = getattr(_mod, "bias", None)
            if b is not None and not isinstance(b, bool): y = y + b
            return y, None
        # vLLM linear layers return (out, bias) unless return_bias False; detect
        rb = getattr(mod, "return_bias", True)
        mod._exp_w8 = (q, s)
        if isinstance(mod, torch.nn.Module) and type(mod).__name__ in ("ReplicatedLinear", "MergedColumnParallelLinear", "QKVParallelLinear", "RowParallelLinear", "ColumnParallelLinear"):
            mod.forward = (lambda xx, _f=fwd, _rb=rb: _f(xx) if _rb else _f(xx)[0])
        else:
            raise RuntimeError(f"unexpected {type(mod).__name__}")
        mod.weight.data = torch.empty(0, dtype=w.dtype, device=w.device)
        tot_b += tb; tot_w += best[0]
        out.append(f"{n}: [{N},{K}] bf16 {tb:.1f}us -> w8 {best[0]:.1f}us cfg={cfg} relerr={err:.4f}")
    out.append(f"sum per draft step: {tot_b:.0f}us -> {tot_w:.0f}us")
    for mg in [r.cudagraph_manager] + [getattr(r.speculator, k) for k in ("prefill_cudagraph_manager", "decode_cudagraph_manager") if getattr(r.speculator, k, None) is not None]:
        mg.graphs.clear()
    gc.collect(); torch.cuda.empty_cache(); r.capture_model(); out.append("recaptured")
else:
    out.append(str([(n, getattr(dm.get_submodule(n), "_exp_w8", None) is not None) for n in names]))
RESULT = "\n".join(out)
