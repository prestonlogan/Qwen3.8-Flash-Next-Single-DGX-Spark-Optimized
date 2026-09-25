#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
"""Long-context prose decode with spec metrics (tok/step, per-pos). usage: t8_long.py N_words [temperature]"""
import json, sys, time, urllib.request, random, re
import os
BASE = "http://127.0.0.1:" + os.environ.get("Q38_PORT", "5810")
N = int(sys.argv[1]); T = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0; random.seed(3)
words = "river mountain village winter market garden letter journey doctor window ocean lantern harvest station silver teacher".split()
sents = []
while len(sents) * 14 < N:
    sents.append(" ".join(random.choice(words) for _ in range(13)) + ".")
msg = " ".join(sents) + "\n\nIgnore the list above. Write an original 500-word short story about a lighthouse keeper who finds a letter from her younger self."
LINE = re.compile(r'^(vllm:[a-z_]+)(\{[^}]*\})? ([0-9.eE+-]+)$')
def met():
    d = {}
    for l in urllib.request.urlopen(BASE + "/metrics").read().decode().splitlines():
        m = LINE.match(l)
        if not m: continue
        k = m.group(1); lab = m.group(2) or ""
        if k == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            k = "pos" + re.search(r'position="(\d+)"', lab).group(1)
        d[k] = d.get(k, 0) + float(m.group(3))
    return d
body = {"model": os.environ.get("MODEL", "qwen3.8-flash-next"), "messages": [{"role": "user", "content": msg}], "max_tokens": 600, "temperature": T, "stream": True,
        "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True}
if T > 0: body.update({"top_p": 0.95, "top_k": 20, "seed": 11})
b = met(); t0 = time.time(); first = None; usage = None
with urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"}), timeout=1800) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]": continue
        d = json.loads(line[5:])
        if d.get("usage"): usage = d["usage"]
        if d.get("choices") and d["choices"][0]["delta"].get("content") and first is None: first = time.time()
t1 = time.time(); a = met(); dd = {k: a.get(k, 0) - b.get(k, 0) for k in a}
dr = dd.get("vllm:spec_decode_num_drafts_total", 0); ct = usage["completion_tokens"]
print(json.dumps({"tag": sys.argv[3] if len(sys.argv) > 3 else "", "T": T, "prompt_tokens": usage["prompt_tokens"], "completion_tokens": ct,
                  "ttft_s": round(first - t0, 1), "decode_tps": round((ct - 1) / (t1 - first), 2),
                  "tok_per_step": round(1 + dd.get("vllm:spec_decode_num_accepted_tokens_total", 0) / dr, 3) if dr else None,
                  "per_pos": [round(dd.get(f"pos{j}", 0) / dr, 3) for j in range(3)] if dr else None}), flush=True)
