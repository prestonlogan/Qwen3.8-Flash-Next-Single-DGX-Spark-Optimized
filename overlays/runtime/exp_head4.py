"""EXPERIMENT: INT4 (sym, group-32, fp16 scale) coarse copy of the full target lm_head for
candidate selection only; exact BF16 logits are recomputed for the top-C candidates."""
import torch
import triton
import triton.language as tl

G = 32


@triton.jit
def _int4_head(X, W, S, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               XM: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in tl.range(0, K, BK, num_stages=3):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * XM + rk[None, :], (rm[:, None] < M), other=0.0)
        rkh = k0 // 2 + tl.arange(0, BK // 2)
        p = tl.load(W + rn[:, None] * (K // 2) + rkh[None, :], (rn[:, None] < N), other=0)
        lo = (p & 0xF).to(tl.int8) - 8
        hi = (p >> 4).to(tl.int8) - 8
        w = tl.reshape(tl.join(lo, hi), (BN, BK)).to(tl.float32)
        rg = k0 // 32 + tl.arange(0, BK // 32)
        s = tl.load(S + rn[:, None] * (K // 32) + rg[None, :], (rn[:, None] < N), other=0.0).to(tl.float32)
        w = tl.reshape(tl.reshape(w, (BN, BK // 32, 32)) * s[:, :, None], (BN, BK))
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
    tl.store(O + rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))


@torch.no_grad()
def quantize_int4_g32(weight: torch.Tensor):
    n, k = weight.shape
    packed = torch.empty((n, k // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((n, k // G), dtype=torch.float16, device=weight.device)
    rows = 8192
    for a in range(0, n, rows):
        c = weight[a:a + rows].float().view(-1, k // G, G)
        s = (c.abs().amax(-1) / 7.0).clamp_min(1e-8)
        q = torch.round(c / s[..., None]).clamp(-8, 7).to(torch.int16) + 8
        q = q.view(-1, k).to(torch.uint8)
        packed[a:a + rows] = q[:, 0::2] | (q[:, 1::2] << 4)
        scales[a:a + rows] = s.to(torch.float16)
    return packed, scales


def int4_logits(h: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, V: int) -> torch.Tensor:
    h = h.to(torch.bfloat16).contiguous()
    m, k = h.shape
    n = packed.shape[0]
    out = torch.empty((m, n), dtype=torch.float32, device=h.device)
    BN = 32
    _int4_head[(triton.cdiv(n, BN),)](h, packed, scales, out, m, n, k, h.stride(0),
                                      max(16, triton.next_power_of_2(m)), BN, 256, num_warps=4)
    return out[:, :V]
