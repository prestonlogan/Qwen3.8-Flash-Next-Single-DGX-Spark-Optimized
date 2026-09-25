# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# E39: exactness gate for the fast (INT2 coarse + BF16 refine) target head.
# The fast head is used only inside runner.sample() for batches where every request is greedy (temperature 0) with
# no penalties / logit bias / allowed ids / min_tokens-stop / bad words / thinking budget / logprobs / logprob token ids,
# and no grammar bitmask. Everything else (sampling, logprobs, guided decoding, prompt logprobs, dummy runs) uses the
# original BF16 head. Mode /exp/head_gate.txt: on | off | stats. No recapture (target head + sampler run eagerly).
import builtins
import numpy as np
r = worker.model_runner; model = r.model
D = builtins.__dict__.setdefault("_exp_hgate", {})
mode = open("/exp/head_gate.txt").read().split()[0]
out = []
def uninstall():
    if "prev_cl" in D:
        model.compute_logits = D.pop("prev_cl"); D.pop("fast", None)
        if D.pop("had_sample_attr"): r.sample = D.pop("prev_sample")
        else: r.__dict__.pop("sample", None); D.pop("prev_sample", None)
        out.append("gate removed")
def make_fast(C):
    import importlib.util, torch
    s_ = importlib.util.spec_from_file_location("exp_head2", "/exp/exp_head2.py"); h2 = importlib.util.module_from_spec(s_); s_.loader.exec_module(h2)
    lm = getattr(model, "lm_head", None) or model.language_model.lm_head; W = lm.weight
    P2, S2 = model._exp_int2; cfg = model._exp_h2cfg; V = P2.shape[0]; orig = model._exp_orig_compute_logits
    def fast_logits(hidden_states, *a, **k):
        m = hidden_states.shape[0]
        if m > 16 or m == 0: return orig(hidden_states, *a, **k)
        h = hidden_states.to(torch.bfloat16).contiguous()
        ap = h2.logits(h, P2, S2, V, *cfg)
        idx = ap.topk(C, dim=1).indices
        w = W.index_select(0, idx.reshape(-1)).view(m, C, -1)
        ex = torch.bmm(w, h.unsqueeze(-1)).squeeze(-1)
        fl = ex.min(dim=1, keepdim=True).values.float()
        o = torch.minimum(ap, fl).to(torch.bfloat16); o.scatter_(1, idx, ex)
        return o
    return fast_logits
if mode == "on":
    uninstall()
    tok = open("/exp/head_gate.txt").read().split()
    fast = model.compute_logits; exact = model._exp_orig_compute_logits
    assert fast is not exact, "fast head not installed"
    D["prev_cl_install"] = fast
    if len(tok) > 1: fast = make_fast(int(tok[1])); out.append(f"fast head rebuilt with C={tok[1]}")
    D["fast"] = fast
    S = r.sampler; ss = S.sampling_states; pen = S.penalties_state; lb = S.logit_bias_state; bw = S.bad_words_state
    tb = S.thinking_budget_state; lti = S.logprob_token_ids_state
    st = D.setdefault("stats", {"fast": 0, "exact_sample": 0, "exact_other": 0})
    flag = {"ok": False}
    def fast_ok(input_batch, grammar_output):
        if grammar_output is not None: return False
        ix = input_batch.idx_mapping_np
        if ix.size == 0: return False
        if not np.all(ss.temperature.np[ix] == 0.0): return False
        if np.any(pen.use_penalty[ix]) or np.any(lb.use_logit_bias[ix]): return False
        if bw.num_bad_words.np[ix].max(initial=0) > 0: return False
        if tb.enabled and np.any(tb.use_thinking_budget[ix]): return False
        if ss.num_logprobs[ix].max(initial=-1) != -1: return False
        if lti.num_token_ids.np[ix].max(initial=0) > 0: return False
        return True
    prev_sample = r.sample
    def sample(hidden_states, input_batch, grammar_output):
        flag["ok"] = fast_ok(input_batch, grammar_output)
        try: return prev_sample(hidden_states, input_batch, grammar_output)
        finally: flag["ok"] = False
    def gated(hidden_states, *a, **k):
        if flag["ok"]:
            st["fast"] += 1; return fast(hidden_states, *a, **k)
        st["exact_other" if hidden_states.shape[0] > 16 else "exact_sample"] += 1
        return exact(hidden_states, *a, **k)
    D["had_sample_attr"] = "sample" in r.__dict__; D["prev_sample"] = prev_sample; D["prev_cl"] = fast
    r.sample = sample; model.compute_logits = gated
    out.append("gate on")
elif mode == "exact":
    uninstall()
    D["prev_cl"] = model.compute_logits; D["had_sample_attr"] = "sample" in r.__dict__; D["prev_sample"] = r.sample
    model.compute_logits = model._exp_orig_compute_logits
    out.append("pure BF16 head everywhere (reference)")
elif mode == "off":
    uninstall()
out.append(f"stats {D.get('stats')}")
RESULT = "\n".join(out)
