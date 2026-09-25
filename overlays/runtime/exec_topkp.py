# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# U30a (v2, stock tie order): exact small-k top-k/top-p masking for sampled batches (idea of vLLM #54651: avoid the full-vocab sort at batch<8).
# Applies only when every row has 1 <= top_k <= KC (=64); otherwise (top_k disabled, k>64, p-only) the original runs.
# Semantics = pinned apply_top_k_top_p_pytorch: top-k threshold (ties at the k-th value kept), then top-p over the
# top-k survivors' softmax, "at least one" kept. Differs only in which of exactly-equal logits the (unstable) sort drops.
# ctl /exp/topkp.txt: on | off | status
import builtins, numpy as np, torch
import vllm.v1.worker.gpu.sample.states as S
C = builtins.__dict__.setdefault("_exp_topkp", {"orig": S.SamplingStates.apply_top_k_top_p, "n_fast": 0, "n_fb": 0})
KC = 64
def _fast(logits, k, p):
    vals, idx = logits.topk(KC, dim=-1)
    # reproduce the stock stable ascending sort order: among equal values, higher index first in descending order
    o = idx.argsort(dim=1, descending=True); vals, idx = vals.gather(1, o), idx.gather(1, o)
    o = vals.argsort(dim=1, descending=True, stable=True); vals, idx = vals.gather(1, o), idx.gather(1, o)
    thr = vals.gather(1, (k.long() - 1)[:, None])
    lv = vals.masked_fill(vals < thr, float("-inf"))
    if p is not None:
        m = vals[:, :1]
        e = torch.exp(lv - m); pr = e / e.sum(-1, keepdim=True)
        asc = pr.flip(1).cumsum(1).flip(1)
        drop = asc <= (1 - p)[:, None]; drop[:, 0] = False
        lv = lv.masked_fill(drop, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter_(1, idx, lv)
def apply_top_k_top_p(self, logits, expanded_idx_mapping, idx_mapping_np):
    kn = self.top_k.np[idx_mapping_np]
    if kn.size and kn.min() >= 1 and kn.max() <= KC and logits.shape[-1] > KC:
        pn = self.top_p.np[idx_mapping_np]
        k = self.top_k.gpu[expanded_idx_mapping]
        p = self.top_p.gpu[expanded_idx_mapping] if np.any(pn != 1.0) else None
        C["n_fast"] += 1
        return _fast(logits, k, p)
    C["n_fb"] += 1
    return C["orig"](self, logits, expanded_idx_mapping, idx_mapping_np)
cmd = open("/exp/topkp.txt").read().split()
if cmd[0] == "on":
    S.SamplingStates.apply_top_k_top_p = apply_top_k_top_p; C["n_fast"] = C["n_fb"] = 0
elif cmd[0] == "off":
    S.SamplingStates.apply_top_k_top_p = C["orig"]
RESULT = f"topkp {cmd} active={S.SamplingStates.apply_top_k_top_p is not C['orig']} fast={C['n_fast']} fallback={C['n_fb']}"
