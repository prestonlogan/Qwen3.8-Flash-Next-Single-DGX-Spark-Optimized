# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# Draft decode graphs at exact sizes (1..4) instead of padding to 4. Draft-only => output-exact under greedy verify.
import gc, torch
r = worker.model_runner; sp = r.speculator; mg = sp.decode_cudagraph_manager
mode = open("/exp/dsize.txt").read().split()[0]
cc = mg.compilation_config
orig = list(cc.cudagraph_capture_sizes)
sizes = [1, 2, 3, 4] if mode == "on" else [4]
cc.cudagraph_capture_sizes = sorted(set(orig) | set(sizes)) if mode == "on" else orig
mg._candidates.clear(); mg._capture_descs.clear() if hasattr(mg._capture_descs, "clear") else None
mg._init_candidates()
cc.cudagraph_capture_sizes = orig
for m in [r.cudagraph_manager, sp.prefill_cudagraph_manager, mg]: m.graphs.clear()
gc.collect(); r.capture_model()
RESULT = f"{mode}: decode keys {[ (d.num_tokens, d.num_reqs) for d in mg.graphs]} ; cands {sorted(mg._candidates)}"
