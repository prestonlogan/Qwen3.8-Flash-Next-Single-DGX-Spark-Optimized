# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# Swap the draft head GEMV to INT4 g32 (exp_head4 kernel) over the same 64,877 reduced rows. Draft-only.
# /exp/draft_head_mode.txt: int4 | fp8
import importlib.util, torch, sys, types
spec = importlib.util.spec_from_file_location("exp_head4", "/exp/exp_head4.py")
h4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(h4)
r = worker.model_runner; m = r.speculator.model
mod = sys.modules[type(m).__module__]
mode = open("/exp/draft_head_mode.txt").read().split()[0]
if not hasattr(mod, "_exp_fp8_logits_orig"):
    mod._exp_fp8_logits_orig = mod._exp_fp8_logits
if mode in ("int4", "int4r", "int4rv"):
    if not hasattr(m, "_exp_draft_int4"):
        tl = getattr(r.model, "lm_head", None) or r.model.language_model.lm_head
        w = tl.weight.index_select(0, m._draft_id_to_target_id.long())
        # sanity: fp8 copy should match these rows
        err = ((m._draft_lm_head_fp8[:512].float() * m._draft_lm_head_fp8_scale[:512, None]) - w[:512].float()).norm() / w[:512].float().norm()
        if err > 0.05: raise RuntimeError(f"row mapping mismatch relerr {err}")
        if w.shape[0] != m._draft_lm_head_fp8.shape[0]: raise RuntimeError(f"draft weight shape {tuple(m._draft_lm_head_weight.shape)} vs fp8 {tuple(m._draft_lm_head_fp8.shape)}")
        m._exp_draft_int4 = h4.quantize_int4_g32(w)
    P, S = m._exp_draft_int4; N = P.shape[0]
    if not hasattr(m, '_exp_draft_cnt') or m._exp_draft_cnt.numel() != 4: m._exp_draft_cnt = torch.zeros(4, dtype=torch.long, device='cuda')
    K = int(open('/exp/draft_head_k.txt').read()) if __import__('os').path.exists('/exp/draft_head_k.txt') else 8
    def int4_logits(x, fp8, scale):
        ap = h4.int4_logits(x, P, S, N)
        if mode not in ("int4r", "int4rv"):
            return ap
        m_ = x.shape[0]
        idx = ap.topk(K, dim=1).indices
        w = fp8.index_select(0, idx.reshape(-1)).view(m_, K, -1).to(torch.bfloat16)
        ex = torch.bmm(w.float(), x.to(torch.bfloat16).float().unsqueeze(-1)).squeeze(-1) * scale[idx]
        out = torch.full_like(ap, -1e30)
        out.scatter_(1, idx, ex)
        if mode == "int4rv":
            ref = mod._exp_fp8_logits_orig(x, fp8, scale)
            cnt = m._exp_draft_cnt
            mx = ref.max(1).values; got = ref.gather(1, out.argmax(1, keepdim=True)).squeeze(1)
            real = (mx.abs() > 0) & (ref.abs().amax(1) > 0)
            cnt[0] += real.sum(); cnt[1] += ((ref.argmax(1) != out.argmax(1)) & real).sum(); cnt[2] += ((mx - got > 1e-3) & real).sum(); cnt[3] += (~real).sum()
        return out
    mod._exp_fp8_logits = int4_logits
else:
    mod._exp_fp8_logits = mod._exp_fp8_logits_orig
# timing
x1 = torch.randn(1, 2560, dtype=torch.bfloat16, device="cuda")
def timeit(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): fn()
    g.replay(); torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / n * 1000
t = timeit(lambda: mod._exp_fp8_logits(x1, m._draft_lm_head_fp8, m._draft_lm_head_fp8_scale))
# agreement of argmax vs fp8 on random-ish real hidden states is not available here; recapture graphs so propose uses new fn
import time, gc
mgrs = [r.cudagraph_manager] + [getattr(r.speculator, k) for k in ("prefill_cudagraph_manager", "decode_cudagraph_manager") if getattr(r.speculator, k, None) is not None]
for mg in mgrs: mg.graphs.clear()
gc.collect(); r.capture_model()
RESULT = f"draft head mode={mode}: M=1 {t:.0f}us; graphs recaptured; cnt={m._exp_draft_cnt.tolist() if hasattr(m, '_exp_draft_cnt') else None}"
