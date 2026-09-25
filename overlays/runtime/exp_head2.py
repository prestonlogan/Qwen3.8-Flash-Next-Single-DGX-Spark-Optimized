# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
"""EXPERIMENT: 2-bit (4-level symmetric, group-16, fp16 scale) coarse copy of the target lm_head, candidate selection only."""
import torch, triton, triton.language as tl
G = 16

@triton.jit
def _int2_head(X, W, S, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               XM: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in tl.range(0, K, BK, num_stages=3):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * XM + rk[None, :], (rm[:, None] < M), other=0.0)
        rkq = k0 // 4 + tl.arange(0, BK // 4)
        p = tl.load(W + rn[:, None] * (K // 4) + rkq[None, :], (rn[:, None] < N), other=0)
        w = tl.reshape(tl.join(tl.join(p & 3, (p >> 2) & 3), tl.join((p >> 4) & 3, p >> 6)), (BN, BK)).to(tl.float32) - 1.5
        rg = k0 // 16 + tl.arange(0, BK // 16)
        s = tl.load(S + rn[:, None] * (K // 16) + rg[None, :], (rn[:, None] < N), other=0.0).to(tl.float32)
        w = tl.reshape(tl.reshape(w, (BN, BK // 16, 16)) * s[:, :, None], (BN, BK))
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
    tl.store(O + rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))

# kernel element order within one byte after join/reshape: [b0-1, b4-5, b2-3, b6-7]
ORDER = (0, 2, 1, 3)   # element j of each 4-group lives in bit-pair ORDER[j]

@torch.no_grad()
def quantize(weight: torch.Tensor):
    n, k = weight.shape
    packed = torch.empty((n, k // 4), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((n, k // G), dtype=torch.float16, device=weight.device)
    for a in range(0, n, 8192):
        f = weight[a:a + 8192].float().view(-1, k // G, G)
        s = (f.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 1.5).half()
        idx = (torch.floor(f / s.float()) + 2).clamp(0, 3).to(torch.uint8).view(-1, k // 4, 4)
        b = torch.zeros(idx.shape[:2], dtype=torch.uint8, device=weight.device)
        for j in range(4):
            b |= idx[..., j] << (2 * ORDER[j])
        packed[a:a + 8192] = b; scales[a:a + 8192] = s.squeeze(-1)
    return packed, scales

def dequant(packed, scales):
    n = packed.shape[0]; k = packed.shape[1] * 4
    idx = torch.stack([(packed >> (2 * ORDER[j])) & 3 for j in range(4)], -1).view(n, k).float() - 1.5
    return (idx.view(n, k // G, G) * scales.float()[..., None]).view(n, k)

def logits(x, P, S, N, BN=64, BK=256, nw=4):
    x = x.to(torch.bfloat16).contiguous(); m, k = x.shape
    out = torch.empty((m, N), dtype=torch.float32, device=x.device)
    _int2_head[(triton.cdiv(N, BN),)](x, P, S, out, m, N, k, x.stride(0), max(16, triton.next_power_of_2(m)), BN, BK, num_warps=nw)
    return out
