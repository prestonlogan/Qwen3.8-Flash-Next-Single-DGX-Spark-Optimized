"""HX: lossless exponent-coded BF16 head. Per weight: byte sign|mantissa7, nibble offset from per-64 group max exponent
(0..14; 15 = escape -> weight coded as 0 and added back by a sparse fp32 correction). Weights reconstruct bit-exactly; logits are NOT bit-identical to cuBLAS (accumulation order; 99.98% equal, TV<=2.4e-6)."""
import torch, triton, triton.language as tl
G = 64
@triton.jit
def _hx(X, SM, OF, GMX, Y, M, N, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0); rn = pid * BN + tl.arange(0, BN); rm = tl.arange(0, BM); nm = rn < N
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in tl.range(0, K, BK, num_stages=3):
        rk = k0 + tl.arange(0, BK)
        s = tl.load(SM + rn[:, None] * K + rk[None, :], nm[:, None], other=0).to(tl.int32)
        p = tl.load(OF + rn[:, None] * (K // 2) + (k0 // 2 + tl.arange(0, BK // 2))[None, :], nm[:, None], other=0).to(tl.int32)
        o = tl.reshape(tl.join(p & 0xF, p >> 4), (BN, BK))
        g = tl.load(GMX + rn[:, None] * (K // 64) + (k0 // 64 + tl.arange(0, BK // 64))[None, :], nm[:, None], other=0).to(tl.int32)
        e = tl.reshape(g[:, :, None] - tl.reshape(o, (BN, BK // 64, 64)), (BN, BK))
        bits = ((s >> 7) << 15) | (e << 7) | (s & 0x7F)
        bits = tl.where(o == 15, 0, bits)
        wv = bits.to(tl.int16).to(tl.bfloat16, bitcast=True)
        x = tl.load(X + rm[:, None] * K + rk[None, :], rm[:, None] < M, other=0.0)
        acc += tl.dot(x, tl.trans(wv), out_dtype=tl.float32)
    tl.store(Y + rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & nm[None, :])
@torch.no_grad()
def compress(W, chunk=8192):
    N, K = W.shape; dev = W.device
    SM = torch.empty(N, K, dtype=torch.uint8, device=dev); OF = torch.empty(N, K // 2, dtype=torch.uint8, device=dev)
    GM = torch.empty(N, K // G, dtype=torch.uint8, device=dev); er, ec, ev = [], [], []
    for a in range(0, N, chunk):
        b = W[a:a + chunk].view(torch.int16).int() & 0xFFFF
        SM[a:a + chunk] = ((((b >> 15) & 1) << 7) | (b & 0x7F)).to(torch.uint8)
        ex = ((b >> 7) & 0xFF).view(-1, K // G, G); gm = ex.amax(-1)
        off = (gm[..., None] - ex).view(-1, K); esc = off >= 15; off = off.clamp(max=15)
        OF[a:a + chunk] = (off[:, 0::2] | (off[:, 1::2] << 4)).to(torch.uint8); GM[a:a + chunk] = gm.to(torch.uint8)
        r, c = esc.nonzero(as_tuple=True); er.append(r + a); ec.append(c); ev.append(W[a:a + chunk][r, c].float())
        del b, ex, off, esc
    return SM, OF, GM, torch.cat(er), torch.cat(ec), torch.cat(ev)
def logits(x, P, cfg=(32, 256, 4)):
    SM, OF, GM, er, ec, ev = P; N, K = SM.shape; M = x.shape[0]; BN, BK, nw = cfg
    y = torch.empty(M, N, dtype=torch.float32, device=x.device)
    _hx[(triton.cdiv(N, BN),)](x, SM, OF, GM, y, M, N, K=K, BM=16, BN=BN, BK=BK, num_warps=nw)
    if er.numel(): y.index_add_(1, er, x.float()[:, ec] * ev[None, :])
    return y
