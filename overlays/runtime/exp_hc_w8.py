# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
"""EXPERIMENT: W8A16 (int8 symmetric, group-64 along K, fp32 scale) linear for the hyper-connection
mixers. Dequantized weight value = bf16(q * s), identical to the fake-quant screened in E13."""
import torch
import triton
import triton.language as tl

G = 64
PDL = False


@triton.jit
def _w8g64(X, W, S, O, M, N, K, XS, OS_SPLIT,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KSPLIT: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    rn = pid_n * BN + tl.arange(0, BN)
    rm = pid_m * BM + tl.arange(0, BM)
    acc = tl.zeros((BM, BN), tl.float32)
    k_per = K // KSPLIT
    k_lo = pid_k * k_per
    for k0 in tl.range(k_lo, k_lo + k_per, BK, num_stages=3):
        rk = k0 + tl.arange(0, BK)
        w = tl.load(W + rn[:, None] * K + rk[None, :], rn[:, None] < N, other=0).to(tl.float32)
        x = tl.load(X + rm[:, None] * XS + rk[None, :], rm[:, None] < M, other=0.0)
        rg = k0 // 64 + tl.arange(0, BK // 64)
        s = tl.load(S + rn[:, None] * (K // 64) + rg[None, :], rn[:, None] < N, other=0.0)
        w = tl.reshape(tl.reshape(w, (BN, BK // 64, 64)) * s[:, :, None], (BN, BK)).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    if KSPLIT == 1:
        tl.store(O + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), msk)
    else:
        tl.store(O + pid_k * OS_SPLIT + rm[:, None] * N + rn[None, :], acc, msk)


@torch.no_grad()
def quantize(w: torch.Tensor):
    n, k = w.shape
    assert k % G == 0
    f = w.float().view(n, k // G, G)
    s = f.abs().amax(-1).clamp_min(1e-12) / 127.0
    q = torch.round(f / s[..., None]).clamp(-127, 127).to(torch.int8).view(n, k)
    return q.contiguous(), s.contiguous()


# per-shape config: (BN, BK, KSPLIT, num_warps) chosen for small M; large M uses BM=64 no split.
CFG = {}


def linear(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, cfg=None) -> torch.Tensor:
    shp = x.shape
    x2 = x.reshape(-1, shp[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m, k = x2.shape
    n = q.shape[0]
    if m <= 16:
        BN, BK, KS, nw = cfg or CFG.get((n, k), (32, 256, 1, 4))
        BM = 16
    else:
        BN, BK, KS, nw = 64, 64, 1, 4
        BM = 64
    grid = (triton.cdiv(n, BN), triton.cdiv(m, BM), KS)
    if KS == 1:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _w8g64[grid](x2, q, s, out, m, n, k, x2.stride(0), 0, BM=BM, BN=BN, BK=BK, KSPLIT=1, num_warps=nw, PDL=PDL, launch_pdl=PDL)
    else:
        part = torch.empty((KS, m, n), dtype=torch.float32, device=x.device)
        _w8g64[grid](x2, q, s, part, m, n, k, x2.stride(0), m * n, BM=BM, BN=BN, BK=BK, KSPLIT=KS, num_warps=nw, PDL=PDL, launch_pdl=PDL)
        out = part.sum(0).to(torch.bfloat16)
    return out.reshape(*shp[:-1], n)
