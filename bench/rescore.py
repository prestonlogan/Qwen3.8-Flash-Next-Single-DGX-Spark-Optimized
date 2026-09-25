#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
"""Distribution check: mean target log-prob of sampled completions, per arm. usage: rescore.py texts.jsonl [prompt_suite]"""
import os, json, sys, http.client, statistics as st, math
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__))); import q38bench as q
def post(path, body):
    c = http.client.HTTPConnection("127.0.0.1", int(os.environ.get("Q38_PORT", "5810")), timeout=600); c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    d = json.loads(c.getresponse().read()); c.close(); return d
M = os.environ.get("MODEL", "qwen3.8-flash-next")
rows = [json.loads(l) for l in open(sys.argv[1])]
res = {}
for r in rows:
    prompts = q.SUITES[r["suite"]] if r["suite"] in q.SUITES else json.load(open(r["suite"])); pr = prompts[r["prompt_idx"]]
    pids = post("/tokenize", {"model": M, "messages": [{"role": "user", "content": pr}], "add_generation_prompt": True,
                              "chat_template_kwargs": {"enable_thinking": False}})["tokens"]
    cids = post("/tokenize", {"model": M, "prompt": r["text"], "add_special_tokens": False})["tokens"]
    d = post("/v1/completions", {"model": M, "prompt": pids + cids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0})
    pl = d["choices"][0]["prompt_logprobs"][len(pids):]
    lps = []
    for tok, ent in zip(cids, pl):
        e = ent.get(str(tok)) if ent else None
        if e is not None: lps.append(e["logprob"])
    res.setdefault(r["tag"], []).append((r["prompt_idx"], st.mean(lps), len(lps)))
for tag, v in sorted(res.items()):
    allm = [x[1] for x in v]
    print(f"{tag}: n_texts={len(v)} tokens={sum(x[2] for x in v)} mean logprob/token={st.mean(allm):.4f} ± {st.stdev(allm)/math.sqrt(len(allm)):.4f} (SE over texts)")
tags = sorted(res)
if len(tags) == 2:
    a, b = tags
    d = [st.mean([x[1] for x in res[b] if x[0] == i]) - st.mean([x[1] for x in res[a] if x[0] == i]) for i in sorted(set(x[0] for x in res[a]) & set(x[0] for x in res[b]))]
    print(f"paired per-prompt diff ({b} - {a}): {st.mean(d):+.4f} ± {st.stdev(d)/math.sqrt(len(d)):.4f} SE, n={len(d)}")
