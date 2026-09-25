# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# One-shot hot install of the current best stack on a fresh E11-style launch (PLE fix + FP8 draft head at boot):
#   1) INT4-coarse(g32) + exact-BF16-refine target head, C=256 (non-verify)
#   2) W8A16 group-64 HC mixers (target + draft), BF16 freed, CUDA graphs recaptured
import importlib.util, time, gc, torch
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
h4 = load("exp_head4", "/exp/exp_head4.py"); W8 = load("exp_hc_w8", "/exp/exp_hc_w8.py")
W8.CFG[(336, 10240)] = (16, 256, 4, 2); W8.CFG[(10240, 320)] = (32, 64, 1, 4)
r = worker.model_runner; model = r.model
out = []
# 1) target head
lm = getattr(model, "lm_head", None) or model.language_model.lm_head
W = lm.weight; V = int(getattr(lm, "org_vocab_size", W.shape[0])); C = 256
if not hasattr(model, "_exp_orig_compute_logits"):
    model._exp_orig_compute_logits = model.compute_logits
    P, S = h4.quantize_int4_g32(W); model._exp_int4 = (P, S)
    orig = model._exp_orig_compute_logits
    def fast_logits(hidden_states, *a, **k):
        m = hidden_states.shape[0]
        if m > 16 or m == 0:
            return orig(hidden_states, *a, **k)
        h = hidden_states.to(torch.bfloat16).contiguous()
        ap = h4.int4_logits(h, P, S, V)
        idx = ap.topk(C, dim=1).indices
        w = W.index_select(0, idx.reshape(-1)).view(m, C, -1)
        exact = torch.bmm(w, h.unsqueeze(-1)).squeeze(-1)
        floor = exact.min(dim=1, keepdim=True).values.float()
        o = torch.minimum(ap, floor).to(torch.bfloat16)
        o.scatter_(1, idx, exact)
        return o
    model.compute_logits = fast_logits
    out.append(f"head: int4 C={C} installed ({(P.numel()+S.numel()*2)/1e9:.2f} GB)")
else:
    out.append("head: already installed")
# 2) HC W8
class W8Lin(torch.nn.Module):
    def __init__(self, w):
        super().__init__(); q, s = W8.quantize(w.data)
        self.register_buffer("q", q, persistent=False); self.register_buffer("s", s, persistent=False)
    def forward(self, x):
        return W8.linear(x, self.q, self.s)
mods = [m for _, m in model.named_modules() if type(m).__name__ == "GatedResidual"]
if getattr(r, "speculator", None) is not None and hasattr(r.speculator, "model"):
    mods += [m for _, m in r.speculator.model.named_modules() if type(m).__name__ == "GatedResidual"]
n = 0
for m in mods:
    for name in ("input_mix_weight_down_block_inject", "input_mix_weight_down", "input_mix_weight_up"):
        lin = getattr(m, name, None)
        if lin is None or type(lin).__name__ == "W8Lin": continue
        setattr(m, name, W8Lin(lin.weight)); n += 1
gc.collect(); torch.cuda.empty_cache()
out.append(f"hc: swapped {n} linears in {len(mods)} modules")
# 3) recapture
mgrs = [r.cudagraph_manager] + [getattr(r.speculator, k) for k in ("prefill_cudagraph_manager", "decode_cudagraph_manager") if getattr(r.speculator, k, None) is not None]
ng = 0
for mg in mgrs:
    ng += len(mg.graphs); mg.graphs.clear()
t0 = time.time(); r.capture_model()
out.append(f"recaptured {sum(len(m.graphs) for m in mgrs)} graphs (cleared {ng}) in {time.time()-t0:.1f}s; cuda free {torch.cuda.mem_get_info()[0]/2**30:.2f} GiB")
RESULT = "\n".join(out)
