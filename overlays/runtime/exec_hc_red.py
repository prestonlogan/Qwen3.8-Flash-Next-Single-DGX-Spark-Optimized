# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# Replace W8 split-K epilogue part.sum(0).to(bf16) (2 ATen kernels) with one Triton reduce+cast kernel. /exp/hc_red.txt on|off
import importlib, sys, torch, triton, triton.language as tl, gc
_lin = next(mm for mm in worker.model_runner.model.modules() if type(mm).__name__ == "W8Lin")
import types
W8 = types.SimpleNamespace(); _g = type(_lin).forward.__globals__["W8"]; W8 = _g
mode = open("/exp/hc_red.txt").read().split()[0]
r = worker.model_runner
@triton.jit
def _red_cast(P, O, MN, KS: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); msk = o < MN
    acc = tl.load(P + o, msk, other=0.0)
    for k in tl.static_range(1, KS):
        acc += tl.load(P + k * MN + o, msk, other=0.0)
    tl.store(O + o, acc.to(tl.bfloat16), msk)
if not hasattr(W8, "_exp_linear0"): W8._exp_linear0 = W8.linear
def linear(x, q, s, cfg=None):
    shp = x.shape; x2 = x.reshape(-1, shp[-1])
    if x2.stride(-1) != 1: x2 = x2.contiguous()
    m, k = x2.shape; n = q.shape[0]
    if m > 16: return W8._exp_linear0(x, q, s, cfg)
    BN, BK, KS, nw = cfg or W8.CFG.get((n, k), (32, 256, 1, 4))
    if KS == 1: return W8._exp_linear0(x, q, s, cfg)
    grid = (triton.cdiv(n, BN), triton.cdiv(m, 16), KS)
    part = torch.empty((KS, m, n), dtype=torch.float32, device=x.device)
    W8._w8g64[grid](x2, q, s, part, m, n, k, x2.stride(0), m * n, BM=16, BN=BN, BK=BK, KSPLIT=KS, num_warps=nw, PDL=W8.PDL, launch_pdl=W8.PDL)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _red_cast[(triton.cdiv(m * n, 1024),)](part, out, m * n, KS=KS, BLOCK=1024, num_warps=4)
    return out.reshape(*shp[:-1], n)
# bitwise check vs original on a real shape
q = s = None
for mm in r.model.modules():
    if type(mm).__name__ == "W8Lin" and mm.q.shape[0] == 336: q, s = mm.q, mm.s; break
x = torch.randn(4, q.shape[1], dtype=torch.bfloat16, device="cuda")
a = W8._exp_linear0(x, q, s); b = linear(x, q, s)
same = torch.equal(a, b); maxd = (a.float() - b.float()).abs().max().item()
W8.linear = linear if mode == "on" else W8._exp_linear0
for mg in [r.cudagraph_manager, r.speculator.prefill_cudagraph_manager, r.speculator.decode_cudagraph_manager]: mg.graphs.clear()
gc.collect(); r.capture_model()
RESULT = f"hc_red {mode}: bitwise_equal={same} maxdiff={maxd:.2e} KS={W8.CFG.get((336, q.shape[1]))}"
