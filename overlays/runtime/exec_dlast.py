# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# Draft prefill: MLP only on last_token rows (other rows' MLP output is never consumed: attention KV is written before
# the MLP, and only hidden[last_token_indices] is sampled / carried). Mode /exp/dlast.txt on|off. Output-exact.
import gc, torch, builtins
r = worker.model_runner; sp = r.speculator; dm = sp.model
mode = open("/exp/dlast.txt").read().split()[0]
layers = [l for n, l in dm.named_modules() if n.startswith("model.layers.") and n.count(".") == 2]
st = builtins.__dict__.setdefault("_exp_dlast", {"on": False, "nreq": 0, "orig": None})
if st["orig"] is None: st["orig"] = sp._prefill
def _prefill(num_reqs, *a, **k):
    st["on"] = True; st["nreq"] = num_reqs
    try: return st["orig"](num_reqs, *a, **k)
    finally: st["on"] = False
for l in layers:
    mlp = l.mlp
    if mode == "on":
        of = type(mlp).forward.__get__(mlp)
        def fwd(x, *a, _of=of, **k):
            if not st["on"] or x.shape[0] <= st["nreq"]: return _of(x, *a, **k)
            idx = sp.last_token_indices[: st["nreq"]]
            y = _of(x.index_select(0, idx), *a, **k)
            out = torch.zeros_like(x) if y.shape[-1] == x.shape[-1] else x.new_zeros(x.shape[0], y.shape[-1])
            out.index_copy_(0, idx, y)
            return out
        mlp.forward = fwd
    else:
        mlp.__dict__.pop("forward", None)
if mode == "on": sp._prefill = _prefill
else: sp.__dict__.pop("_prefill", None)
for m in [r.cudagraph_manager, sp.prefill_cudagraph_manager, sp.decode_cudagraph_manager]: m.graphs.clear()
gc.collect(); r.capture_model()
RESULT = f"dlast {mode} on {len(layers)} draft layers; decode keys {[(d.num_tokens) for d in sp.decode_cudagraph_manager.graphs]}"
