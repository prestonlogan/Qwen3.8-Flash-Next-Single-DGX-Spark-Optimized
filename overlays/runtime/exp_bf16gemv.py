# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
import torch, triton, triton.language as tl
@triton.jit
def _bg(X, W, O, M, N, K, XS, OS_SPLIT, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KSPLIT: tl.constexpr, OUT_F32: tl.constexpr):
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN); rm = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), tl.float32)
    k_per = K // KSPLIT; k_lo = pid_k * k_per
    for k0 in tl.range(k_lo, k_lo + k_per, BK, num_stages=3):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * XS + rk[None, :], rm[:, None] < M, other=0.0)
        w = tl.load(W + rn[:, None] * K + rk[None, :], rn[:, None] < N, other=0.0)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    if KSPLIT == 1:
        if OUT_F32:
            tl.store(O + rm[:, None] * N + rn[None, :], acc, msk)
        else:
            tl.store(O + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), msk)
    else:
        tl.atomic_add(O + rm[:, None] * N + rn[None, :], acc, msk, sem="relaxed")

def linear(x, w, cfg, out_dtype=torch.bfloat16):
    BN, BK, KS, nw = cfg
    x2 = x.reshape(-1, x.shape[-1])
    if x2.stride(-1) != 1: x2 = x2.contiguous()
    m, k = x2.shape; n = w.shape[0]
    if KS == 1:
        out = torch.empty((m, n), dtype=out_dtype, device=x.device)
        _bg[(triton.cdiv(n, BN), 1)](x2, w, out, m, n, k, x2.stride(0), 0, BM=16, BN=BN, BK=BK, KSPLIT=1, OUT_F32=out_dtype == torch.float32, num_warps=nw)
    else:
        acc = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        _bg[(triton.cdiv(n, BN), KS)](x2, w, acc, m, n, k, x2.stride(0), 0, BM=16, BN=BN, BK=BK, KSPLIT=KS, OUT_F32=True, num_warps=nw)
        out = acc if out_dtype == torch.float32 else acc.to(out_dtype)
    return out.view(*x.shape[:-1], n)
