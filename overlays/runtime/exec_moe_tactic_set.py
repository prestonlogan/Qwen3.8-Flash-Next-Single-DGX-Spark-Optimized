# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# Override FlashInfer fused-MoE tactics for M=4 decode; recapture. /exp/moe_tactic.txt: "<g1> <g2>" or "orig"
import ast, gc
from flashinfer.autotuner import AutoTuner
import builtins
r = worker.model_runner; AT = AutoTuner.get()
arg = open("/exp/moe_tactic.txt").read().split()
orig = builtins.__dict__.setdefault("_exp_moe_tactic_orig", {k: v for k, v in AT._file_configs.items() if "fused_moe" in k})
out = []
for k, v in orig.items():
    kk = ast.literal_eval(k)
    if arg[0] == "orig" or kk[2][0][0] != 4:
        AT._file_configs[k] = v
    else:
        t = int(arg[0]) if "gemm1" in kk[0] else int(arg[1])
        AT._file_configs[k] = (v[0], t); out.append(f"{kk[0]} M=4: {v[1]} -> {t}")
for m in [r.cudagraph_manager, r.speculator.prefill_cudagraph_manager, r.speculator.decode_cudagraph_manager]: m.graphs.clear()
gc.collect(); r.capture_model()
RESULT = "; ".join(out) or "orig"
