# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
"""W8A16 GEMV that reads MXFP8 weights exactly (fp8 e4m3 + e8m0 per-32 scale, unswizzled copy), BF16 activations."""
import torch, triton, triton.language as tl

@triton.jit
def _mx8(X, W, S, O, M, N, K, XS, OS_SPLIT, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KSPLIT: tl.constexpr):
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN); rm = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), tl.float32)
    k_per = K // KSPLIT; k_lo = pid_k * k_per
    for k0 in tl.range(k_lo, k_lo + k_per, BK, num_stages=3):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * XS + rk[None, :], rm[:, None] < M, other=0.0)
        w = tl.load(W + rn[:, None] * K + rk[None, :], rn[:, None] < N, other=0.0).to(tl.float32)
        rg = k0 // 32 + tl.arange(0, BK // 32)
        e = tl.load(S + rn[:, None] * (K // 32) + rg[None, :], rn[:, None] < N, other=127).to(tl.int32)
        sc = tl.exp2((e - 127).to(tl.float32))
        w = tl.reshape(tl.reshape(w, (BN, BK // 32, 32)) * sc[:, :, None], (BN, BK)).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    if KSPLIT == 1:
        tl.store(O + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), msk)
    else:
        tl.store(O + pid_k * OS_SPLIT + rm[:, None] * N + rn[None, :], acc, msk)

def unswizzle(flat, N, K):
    mt, kt = (N + 127) // 128, (K + 127) // 128
    return flat.view(torch.uint8).view(mt, kt, 32, 4, 4).transpose(1, 3).reshape(mt * 128, kt * 4)[:N, :K // 32].contiguous()

def linear(x, w8, s, cfg):
    x2 = x.reshape(-1, x.shape[-1])
    if x2.stride(-1) != 1: x2 = x2.contiguous()
    m, k = x2.shape; n = w8.shape[0]
    BN, BK, KS, nw = cfg
    if KS == 1:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _mx8[(triton.cdiv(n, BN), 1)](x2, w8, s, out, m, n, k, x2.stride(0), 0, BM=16, BN=BN, BK=BK, KSPLIT=1, num_warps=nw)
    else:
        part = torch.empty((KS, m, n), dtype=torch.float32, device=x.device)
        _mx8[(triton.cdiv(n, BN), KS)](x2, w8, s, part, m, n, k, x2.stride(0), m * n, BM=16, BN=BN, BK=BK, KSPLIT=KS, num_warps=nw)
        out = part.sum(0).to(torch.bfloat16)
    return out.view(*x.shape[:-1], n)
