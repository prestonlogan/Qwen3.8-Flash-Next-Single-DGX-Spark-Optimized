#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
"""q38bench — isolated decode benchmark for the Qwen3.8 Flash Next decode-speed experiment.

Mirrors the sparkDash DecodeBench protocol (temperature 0, top_p 1, thinking off,
ignore_eos, decode tok/s = (completion_tokens - 1) / (t_last - t_first)) and adds
/metrics deltas (engine ms/step, committed tokens/step, per-position acceptance).

Primary gate: PROSE set, 1 stream. Secondary: code / json / sampled prose / 2-stream.

Usage:
  python3 q38bench.py --port 5810 --tag BASE --suite prose --streams 1 --out results/x.jsonl
"""
import argparse, http.client, json, re, statistics, sys, threading, time, urllib.request

PROSE = [
    # 0: exact sparkDash prose prompt (anchor for the 48.7 tok/s baseline)
    "Write a detailed step-by-step explanation of how a hash map works, "
    "including collision handling, resizing, and time complexity. Be thorough.",
    "Write a long, vivid short story about a lighthouse keeper on a remote island who discovers "
    "that the light has been signalling to something that is not a ship. Use rich description and dialogue.",
    "Write an in-depth historical essay on the causes and consequences of the collapse of the Bronze Age "
    "civilizations in the eastern Mediterranean around 1200 BCE. Discuss competing theories.",
    "Write a thoughtful, conversational letter to a friend who is considering leaving a stable job to start "
    "a small bakery. Weigh the risks and rewards honestly and share practical advice.",
    "Explain to an intelligent non-specialist how the human immune system distinguishes self from non-self, "
    "covering innate and adaptive immunity, T cells, B cells, and what goes wrong in autoimmune disease.",
    "Write a detailed travel guide for someone spending a rainy week in Edinburgh in November: neighbourhoods, "
    "food, museums, day trips, and how to make the most of the weather.",
]
CODE = [
    "Write a complete, well-structured Python command-line tool that parses an nginx access log, "
    "aggregates requests per status code, per endpoint, and per hour, and prints a summary report. "
    "Include argument parsing, error handling, and type hints.",
    "Implement a thread-safe LRU cache in C++17 with a fixed capacity, supporting get, put, and erase, "
    "plus a small test program. Explain nothing; output only code.",
]
JSON = [
    "Write only valid JSON (no markdown). Produce a detailed inventory for a fictional hardware store: an array "
    "of 40 products, each with id, name, category, price, stock, supplier {name, country}, and tags. Vary the values.",
]
STRUCT = ["Count from 1 to 200. Output only the numbers, separated by spaces. No other text."]
# PROSE2: sealed T8 confirmation set (written 2026-09-24 before any use; not used for training, tuning or checkpoint
# selection). Styles and topics deliberately disjoint from bench/gen_corpus.py TASKS x TOPICS and from PROSE.
PROSE2 = [
    "Write a long first-person account by a night-shift nurse in a busy city hospital describing one unusually quiet night and what she noticed.",
    "Explain in depth how GPS satellites let a phone determine its position, including timing, relativity corrections, and sources of error.",
    "Write a detailed review of an imaginary mid-range electric bicycle after six months of daily commuting: comfort, range, repairs, and value.",
    "Write a warm, candid speech a retiring high-school science teacher gives at the final assembly of her career.",
    "Describe, as a narrative history, how the Dutch reclaimed land from the sea over several centuries and what it cost them.",
    "Write a long short story about two strangers stuck overnight in a small regional airport during a snowstorm.",
    "Explain how modern passenger elevators are kept safe: cables, overspeed governors, safety brakes, and inspections.",
    "Write an analytical essay on why some cities become known for a single dish, using several real examples.",
    "Write a detailed account of how a professional orchestra prepares a new symphony in the week before its premiere.",
    "Write a reflective letter from a grandfather to his newborn granddaughter about the world she is being born into.",
    "Explain how noise-cancelling headphones work, from microphones to anti-phase signals, and why they struggle with voices.",
    "Write a vivid description of a traditional night market in Taipei, following one visitor from dusk until closing.",
]
SUITES = {"prose2": PROSE2, "prose": PROSE, "code": CODE, "json": JSON, "structured": STRUCT, "anchor": PROSE[:1]}

COUNTERS = {
    "vllm:inter_token_latency_seconds_sum", "vllm:inter_token_latency_seconds_count",
    "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total", "vllm:generation_tokens_total",
}
LINE = re.compile(r'^(vllm:[a-z_]+)(\{[^}]*\})? ([0-9.eE+-]+)$')


def metrics(port):
    out = {}
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as r:
        for raw in r.read().decode().splitlines():
            m = LINE.match(raw)
            if not m:
                continue
            name, labels, val = m.groups()
            if name in COUNTERS:
                out[name] = out.get(name, 0.0) + float(val)
            elif name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
                pos = re.search(r'position="(\d+)"', labels or "").group(1)
                out[f"pos{pos}"] = out.get(f"pos{pos}", 0.0) + float(val)
    return out


def model_id(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as r:
        return json.loads(r.read())["data"][0]["id"]


TOP_P = None
TOP_K = None
def stream_one(port, model, prompt, max_tokens, temperature, seed, res, idx):
    body = {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature, "top_p": 1.0 if temperature == 0 else (TOP_P if TOP_P is not None else 0.95),
        "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if temperature > 0:
        body["top_k"] = TOP_K if TOP_K is not None else 20
        body["seed"] = seed
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    first = last = None
    n_chunks = 0
    usage = None
    text = []
    buf = b""
    while True:
        chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(1024)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "") + (delta.get("reasoning") or "")
                if piece:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    n_chunks += 1
                    text.append(piece)
    conn.close()
    ct = (usage or {}).get("completion_tokens", 0)
    dec = (ct - 1) / (last - first) if first and last and last > first and ct > 1 else 0.0
    res[idx] = {"ttft_ms": (first - t0) * 1000 if first else None, "decode_tps": dec,
                "completion_tokens": ct, "wall_s": time.perf_counter() - t0, "chunks": n_chunks,
                "t_first": first, "t_last": last, "text": "".join(text)}


def run_wave(port, model, prompts, max_tokens, temperature, seed):
    res = [None] * len(prompts)
    ths = [threading.Thread(target=stream_one, args=(port, model, p, max_tokens, temperature, seed + i, res, i))
           for i, p in enumerate(prompts)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    firsts = [r["t_first"] for r in res if r["t_first"]]
    lasts = [r["t_last"] for r in res if r["t_last"]]
    toks = sum(max(r["completion_tokens"] - 1, 0) for r in res)
    agg = toks / (max(lasts) - min(firsts)) if firsts and lasts else 0.0
    return res, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5810)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--suite", nargs="+", default=["prose"])
    ap.add_argument("--streams", type=int, nargs="+", default=[1])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--texts", default="", help="optional jsonl path to save generated texts (for correctness diffs)")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args()
    global TOP_P, TOP_K
    TOP_P, TOP_K = a.top_p, a.top_k
    model = model_id(a.port)
    if not a.no_warmup:
        run_wave(a.port, model, [STRUCT[0]], 32, 0.0, 0)
    hdr = f"{'tag':<10}{'suite':<11}{'i':>3}{'S':>3}{'rep':>4}{'tok/s':>8}{'agg':>8}{'ms/step':>9}{'tok/step':>9}  p1/p2/p3"
    print(hdr, flush=True)
    for rep in range(a.repeats):
        for s in a.streams:
            for suite in a.suite:
                prompts = SUITES[suite] if suite in SUITES else json.load(open(suite))   # suite may be a JSON prompt-list file
                for i, p in enumerate(prompts):
                    wave_prompts = [prompts[(i + k) % len(prompts)] for k in range(s)]
                    before = metrics(a.port)
                    res, agg = run_wave(a.port, model, wave_prompts, a.max_tokens, a.temperature, a.seed)
                    after = metrics(a.port)
                    d = {k: after.get(k, 0) - before.get(k, 0) for k in set(before) | set(after)}
                    drafts = d.get("vllm:spec_decode_num_drafts_total", 0)
                    itl_n = d.get("vllm:inter_token_latency_seconds_count", 0)
                    row = {
                        "tag": a.tag, "suite": suite, "prompt_idx": i, "S": s, "rep": rep,
                        "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "max_tokens": a.max_tokens,
                        "temperature": a.temperature,
                        "decode_tps_mean": statistics.mean(r["decode_tps"] for r in res),
                        "aggregate_tps": agg,
                        "ttft_ms_mean": statistics.mean(r["ttft_ms"] or 0 for r in res),
                        "completion_tokens": [r["completion_tokens"] for r in res],
                        "ms_per_step": d.get("vllm:inter_token_latency_seconds_sum", 0) / itl_n * 1000 if itl_n else None,
                        "tok_per_step": 1 + d.get("vllm:spec_decode_num_accepted_tokens_total", 0) / drafts if drafts else 1.0,
                        "per_pos": [d.get(f"pos{j}", 0) / drafts if drafts else 0.0 for j in range(8) if f"pos{j}" in d],
                        "drafts": drafts, "note": a.note,
                    }
                    with open(a.out, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    if a.texts:
                        with open(a.texts, "a") as f:
                            for k, r in enumerate(res):
                                f.write(json.dumps({"tag": a.tag, "suite": suite, "prompt_idx": (i + k) % len(prompts),
                                                    "S": s, "rep": rep, "temperature": a.temperature,
                                                    "text": r["text"]}) + "\n")
                    pp = "/".join(f"{x:.2f}" for x in row["per_pos"][:5])
                    print(f"{a.tag:<10}{suite:<11}{i:>3}{s:>3}{rep:>4}{row['decode_tps_mean']:>8.2f}{agg:>8.2f}"
                          f"{(row['ms_per_step'] or 0):>9.2f}{row['tok_per_step']:>9.3f}  {pp}", flush=True)


if __name__ == "__main__":
    main()
