# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
"""W6A16 g32 GEMV: q in [-32,31] stored as lo-nibble plane [N,K/2] + hi-2bit plane [N,K/4], fp16 scale per 32."""
import torch, triton, triton.language as tl
G = 32
PDL = False

@triton.jit
def _w6(X, L, H, S, O, M, N, K, XS, OS_SPLIT, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KSPLIT: tl.constexpr, PDL: tl.constexpr = False):
    pid_n = tl.program_id(0); pid_m = tl.program_id(1); pid_k = tl.program_id(2)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    rn = pid_n * BN + tl.arange(0, BN); rm = pid_m * BM + tl.arange(0, BM)
    acc = tl.zeros((BM, BN), tl.float32)
    k_per = K // KSPLIT; k_lo = pid_k * k_per
    for k0 in tl.range(k_lo, k_lo + k_per, BK, num_stages=3):
        rk = k0 + tl.arange(0, BK)
        rl = k0 // 2 + tl.arange(0, BK // 2)
        p = tl.load(L + rn[:, None] * (K // 2) + rl[None, :], rn[:, None] < N, other=0)
        x = tl.load(X + rm[:, None] * XS + rk[None, :], rm[:, None] < M, other=0.0)
        lo = tl.reshape(tl.join(p & 0xF, p >> 4), (BN, BK)).to(tl.int32)
        rh = k0 // 4 + tl.arange(0, BK // 4)
        h = tl.load(H + rn[:, None] * (K // 4) + rh[None, :], rn[:, None] < N, other=0)
        hi = tl.reshape(tl.join(tl.join(h & 3, (h >> 2) & 3), tl.join((h >> 4) & 3, h >> 6)), (BN, BK)).to(tl.int32)
        q = (lo | (hi << 4)) - 32
        rg = k0 // 32 + tl.arange(0, BK // 32)
        s = tl.load(S + rn[:, None] * (K // 32) + rg[None, :], rn[:, None] < N, other=0.0).to(tl.float32)
        w = tl.reshape(tl.reshape(q.to(tl.float32), (BN, BK // 32, 32)) * s[:, :, None], (BN, BK)).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    if KSPLIT == 1:
        tl.store(O + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), msk)
    else:
        tl.store(O + pid_k * OS_SPLIT + rm[:, None] * N + rn[None, :], acc, msk)

@torch.no_grad()
def quantize(w: torch.Tensor):
    """w: [N,K] float weights -> (lo [N,K/2] u8, hi [N,K/4] u8, s [N,K/32] f16). Also returns dequant ref fn."""
    n, k = w.shape
    L = torch.empty((n, k // 2), dtype=torch.uint8, device=w.device)
    H = torch.empty((n, k // 4), dtype=torch.uint8, device=w.device)
    S = torch.empty((n, k // G), dtype=torch.float16, device=w.device)
    for a in range(0, n, 2048):
        f = w[a:a + 2048].float().view(-1, k // G, G)
        s = (f.abs().amax(-1, keepdim=True) / 31.0).to(torch.float16).float().clamp_min(6.2e-5)
        q = (torch.round(f / s).clamp(-32, 31).to(torch.int32) + 32).view(-1, k)
        lo = q & 15; hi = q >> 4
        L[a:a + 2048] = (lo[:, 0::2] | (lo[:, 1::2] << 4)).to(torch.uint8)
        # kernel decode order: join(join(f0,f1), join(f2,f3)) -> position 4i+2b+a holds field (2a+b)
        # field0=bits0-1, field1=bits2-3, field2=bits4-5, field3=bits6-7
        # pos0<-f0, pos1<-f2, pos2<-f1, pos3<-f3
        H[a:a + 2048] = (hi[:, 0::4] | (hi[:, 2::4] << 2) | (hi[:, 1::4] << 4) | (hi[:, 3::4] << 6)).to(torch.uint8)
        S[a:a + 2048] = s.squeeze(-1).to(torch.float16)
    return L, H, S

def dequant(L, H, S):
    n = L.shape[0]; k = L.shape[1] * 2
    lo = torch.stack((L & 15, L >> 4), -1).view(n, k).int()
    hs = torch.stack((H & 3, (H >> 4) & 3, (H >> 2) & 3, H >> 6), -1).view(n, k).int()
    q = (lo | (hs << 4)) - 32
    return (q.float().view(n, k // 32, 32) * S.float()[..., None]).view(n, k)

def linear(x, L, H, S, cfg):
    x2 = x.reshape(-1, x.shape[-1])
    if x2.stride(-1) != 1: x2 = x2.contiguous()
    m, k = x2.shape; n = L.shape[0]
    if m <= 16:
        BN, BK, KS, nw = cfg; BM = 16
    else:
        BN, BK, KS, nw = 64, 64, 1, 4; BM = 64
    grid = (triton.cdiv(n, BN), triton.cdiv(m, BM), KS)
    if KS == 1:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _w6[grid](x2, L, H, S, out, m, n, k, x2.stride(0), 0, BM=BM, BN=BN, BK=BK, KSPLIT=1, num_warps=nw, PDL=PDL, launch_pdl=PDL)
    else:
        part = torch.empty((KS, m, n), dtype=torch.float32, device=x.device)
        _w6[grid](x2, L, H, S, part, m, n, k, x2.stride(0), m * n, BM=BM, BN=BN, BK=BK, KSPLIT=KS, num_warps=nw, PDL=PDL, launch_pdl=PDL)
        out = part.sum(0).to(torch.bfloat16)
    return out.view(*x.shape[:-1], n)
