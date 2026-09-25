# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# Target head coarse = int2 g16 (C from /exp/head2.txt "on C" | "test" | "int4").
import importlib.util, torch, gc, itertools
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
h2 = load("exp_head2", "/exp/exp_head2.py"); h4 = load("exp_head4", "/exp/exp_head4.py")
r = worker.model_runner; model = r.model
lm = getattr(model, "lm_head", None) or model.language_model.lm_head
W = lm.weight; V = int(getattr(lm, "org_vocab_size", W.shape[0]))
arg = open("/exp/head2.txt").read().split()
out = []
if not hasattr(model, "_exp_int2"):
    model._exp_int2 = h2.quantize(W[:V])
P2, S2 = model._exp_int2
def timeit(fn, n=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): fn()
    g.replay(); torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / n * 1000
# correctness vs dequant reference on first 4096 rows
x = torch.randn(4, 2560, dtype=torch.bfloat16, device="cuda")
ref = torch.nn.functional.linear(x.float(), h2.dequant(P2[:4096], S2[:4096]))
got = h2.logits(x, P2[:4096], S2[:4096], 4096)
out.append(f"int2 kernel relerr vs dequant: {((got-ref).norm()/ref.norm()).item():.2e}; bytes {(P2.numel()+S2.numel()*2)/1e6:.0f} MB")
P4, S4 = model._exp_int4
out.append(f"int4 coarse M=4: {timeit(lambda: h4.int4_logits(x, P4, S4, V)):.0f} us")
best = None
for BN, BK, nw in itertools.product((32, 64, 128), (128, 256, 512), (4, 8)):
    try: t = timeit(lambda: h2.logits(x, P2, S2, V, BN, BK, nw))
    except Exception: continue
    if best is None or t < best[0]: best = (t, BN, BK, nw)
out.append(f"int2 coarse M=4 best {best[0]:.0f} us cfg {best[1:]}")
model._exp_h2cfg = best[1:]
if arg[0] in ("on", "int4"):
    C = int(arg[1]) if len(arg) > 1 else 256
    orig = model._exp_orig_compute_logits
    use2 = arg[0] == "on"
    VER = "verify" in arg
    if not hasattr(model, "_exp_h2cnt"): model._exp_h2cnt = torch.zeros(3, dtype=torch.long, device="cuda")
    cnt = model._exp_h2cnt
    def fast_logits(hidden_states, *a, **k):
        m = hidden_states.shape[0]
        if m > 16 or m == 0:
            return orig(hidden_states, *a, **k)
        h = hidden_states.to(torch.bfloat16).contiguous()
        ap = h2.logits(h, P2, S2, V, *model._exp_h2cfg) if use2 else h4.int4_logits(h, P4, S4, V)
        idx = ap.topk(C, dim=1).indices
        w = W.index_select(0, idx.reshape(-1)).view(m, C, -1)
        exact = torch.bmm(w, h.unsqueeze(-1)).squeeze(-1)
        floor = exact.min(dim=1, keepdim=True).values.float()
        o = torch.minimum(ap, floor).to(torch.bfloat16)
        o.scatter_(1, idx, exact)
        if VER:
            ref = orig(hidden_states, *a, **k)
            rf = ref.float(); t1 = rf.argmax(1); got = o.float().argmax(1)
            gap = rf.max(1).values - rf.gather(1, got[:, None]).squeeze(1)
            cnt[0] += m; cnt[1] += (t1 != got).sum(); cnt[2] += (gap > 0).sum()
            return ref
        return o
    model.compute_logits = fast_logits
    out.append(f"head: {'int2' if use2 else 'int4'} coarse C={C}; full head M=4 {timeit(lambda: fast_logits(x)):.0f} us")
    for mg in [r.cudagraph_manager, r.speculator.prefill_cudagraph_manager, r.speculator.decode_cudagraph_manager]: mg.graphs.clear()
    gc.collect(); r.capture_model(); out.append("recaptured")
out.append(f"cnt(rows, argmax diff, diff with gap>0)={model._exp_h2cnt.tolist() if hasattr(model, '_exp_h2cnt') else None}")
RESULT = "\n".join(out)
