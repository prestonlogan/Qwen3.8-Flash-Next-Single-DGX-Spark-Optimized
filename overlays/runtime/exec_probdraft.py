# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# S2: probabilistic MTP drafting with the fast reduced-vocab draft head. /exp/probdraft.txt: "on K TOPP MULT [indep]" | "off" | "status"
# "indep" (CK1 fix): draw the draft Gumbel noise from a key disjoint from the verification keys. The pinned V2 sampler
# keys draft noise at pos+1, which equals the rejection sampler's key for the residual resample of the same row; that
# coupling biases the output distribution (TV up to ~0.045 per token in exec_coupling2). Independent keys are exact.
# Draft logits = fast head (INT2 coarse + exact refine, the live E34 path), top-K kept, scattered into a full-vocab BF16 row
# (-inf elsewhere). vLLM's gumbel_sample caches exactly these pre-temperature logits and its rejection sampler uses them
# as q, so sampled output stays exactly target-distributed for any q. Greedy requests: argmax unchanged.
import torch, gc, sys, builtins
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
r = worker.model_runner; sp = r.speculator; m = sp.model
mod = sys.modules[type(m).__module__]
arg = open("/exp/probdraft.txt").read().split()
ST = builtins.__dict__.setdefault("_exp_probdraft", {})
out = []
V = int(sp.vocab_size)
dv = m._draft_id_to_target_id.long()

def recapture():
    for mg in [r.cudagraph_manager, sp.prefill_cudagraph_manager, sp.decode_cudagraph_manager]:
        if mg is not None: mg.graphs.clear()
    gc.collect(); torch.cuda.empty_cache(); r.capture_model()

if arg[0] == "on":
    K = int(arg[1]) if len(arg) > 1 else 20
    TOPP = float(arg[2]) if len(arg) > 2 else 1.0      # draft-side top-p (on softmax(v/T)); any q is exact
    MULT = float(arg[3]) if len(arg) > 3 else 1.0      # draft temperature multiplier: q = softmax(v/(T*MULT))
    KEYOFF = 0x5DEECE66 if (len(arg) > 4 and arg[4] == "indep") else 0   # < 2**31 - max_model_len, never collides with a row key
    if sp.draft_logits is None:
        sp.draft_logits = torch.zeros(sp.max_num_reqs, sp.num_speculative_steps, V, dtype=torch.bfloat16, device="cuda")
    def fast_full_logits(h, temperature, idx_mapping):
        x = h.to(torch.bfloat16)
        o = mod._exp_fp8_logits(x, m._draft_lm_head_fp8, m._draft_lm_head_fp8_scale)   # [M, Vd] fp32, exact on refine set
        b = getattr(m, "_exp_draft_bias", None)
        if b is not None: o = o + b
        v, i = o.topk(K, dim=1)                                   # sorted desc
        if MULT != 1.0: v = v * (1.0 / MULT)
        if TOPP < 1.0:
            T = temperature[idx_mapping.clamp(min=0).long()].float().clamp(min=1e-4)[:, None]
            pk = torch.softmax(v / T, dim=1); cs = pk.cumsum(1)
            v = torch.where((cs - pk) < TOPP, v, torch.full_like(v, float("-inf")))
        full = torch.full((x.shape[0], V), float("-inf"), dtype=torch.bfloat16, device=x.device)
        full.scatter_(1, dv[i], v.to(torch.bfloat16))
        return full
    def sample_draft(hidden_states, positions, idx_mapping, temperature, seeds, draft_step, draft_logits):
        return gumbel_sample(fast_full_logits(hidden_states, temperature, idx_mapping), idx_mapping, temperature, seeds, positions + 1 + KEYOFF,
                             apply_temperature=True, logits_cache=draft_logits, logits_cache_col=draft_step,
                             use_fp64=sp.use_fp64_gumbel)
    sp.sample_draft = sample_draft
    ST["on"] = (K, TOPP, MULT, KEYOFF)
    recapture()
    out.append(f"probdraft ON K={K} topp={TOPP} mult={MULT} keyoff={KEYOFF:#x}; draft_logits {tuple(sp.draft_logits.shape)} {sp.draft_logits.dtype}")
elif arg[0] == "off":
    if "sample_draft" in sp.__dict__: del sp.__dict__["sample_draft"]
    sp.draft_logits = None; ST.pop("on", None)
    recapture()
    out.append("probdraft OFF (greedy local-argmax draft restored)")
else:
    out.append(f"status on={ST.get('on')} draft_logits={None if sp.draft_logits is None else tuple(sp.draft_logits.shape)} override={'sample_draft' in sp.__dict__}")
RESULT = "\n".join(out)
