# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
import json, sys, statistics as st
R = [json.loads(l) for l in open(sys.argv[1])]
tags = sorted(set(r["tag"] for r in R))
for t in tags:
    X = [r for r in R if r["tag"] == t]
    print("%-10s n=%2d tps=%.2f tok/step=%.3f ms=%.2f" % (t, len(X), st.mean(r["decode_tps_mean"] for r in X), st.mean(r["tok_per_step"] for r in X), st.mean(r["ms_per_step"] for r in X)))
if len(tags) == 2:
    a, b = [t for t in tags if t.startswith("A_") or t == "r32a"] or tags[:1], None
    A = ("off" if "off" in tags else [t for t in tags if t.startswith("base")][0] if any(t.startswith("base") for t in tags) else (tags[0] if "r32a" in tags[0] else tags[1])); B = [t for t in tags if t != A][0]
    g = []
    for i in sorted(set(r["prompt_idx"] for r in R)):
        x = [r["tok_per_step"] for r in R if r["tag"] == B and r["prompt_idx"] == i]; y = [r["tok_per_step"] for r in R if r["tag"] == A and r["prompt_idx"] == i]
        if x and y: g.append(st.mean(x) / st.mean(y) - 1)
    ta = st.mean(r["tok_per_step"] for r in R if r["tag"] == A); tb = st.mean(r["tok_per_step"] for r in R if r["tag"] == B)
    pa = st.mean(r["decode_tps_mean"] for r in R if r["tag"] == A); pb = st.mean(r["decode_tps_mean"] for r in R if r["tag"] == B)
    se = st.stdev(g) / len(g) ** 0.5 if len(g) > 1 else 0
    print(f"{B} vs {A}: tok/step {100*(tb/ta-1):+.2f}%  tps {100*(pb/pa-1):+.2f}%  per-prompt mean {100*st.mean(g):+.2f}% ± {100*se:.2f} (SE)  positive {sum(x>0 for x in g)}/{len(g)}")
