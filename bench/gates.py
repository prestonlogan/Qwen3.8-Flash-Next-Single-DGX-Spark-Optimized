# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
"""Functional gates: tool calling, strict JSON, reasoning (thinking on), code exec, multi-turn. Greedy.
usage: gates.py TAG -> results/gates_TAG.json"""
import json, sys, urllib.request, re, subprocess, time, os
URL = os.environ.get("URL", "http://127.0.0.1:" + os.environ.get("Q38_PORT", "5810")); M = os.environ.get("MODEL", "qwen3.8-flash-next")
def chat(msgs, think=False, tools=None, max_tokens=1200, **kw):
    b = {"model": M, "messages": msgs, "max_tokens": max_tokens, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": think}}
    if tools: b["tools"] = tools; b["tool_choice"] = "auto"
    b.update(kw)
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(b).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))["choices"][0]["message"]
res = {}
W = {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}
S = {"type": "function", "function": {"name": "search_flights", "description": "Search flights",
     "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"}}, "required": ["origin", "destination", "date"]}}}
# 1 tool call
m = chat([{"role": "user", "content": "What's the weather in Paris in celsius?"}], tools=[W, S])
tc = m.get("tool_calls") or []
ok = len(tc) == 1 and tc[0]["function"]["name"] == "get_weather" and json.loads(tc[0]["function"]["arguments"]).get("city", "").lower().startswith("paris")
res["tool_single"] = {"ok": ok, "calls": [(t["function"]["name"], t["function"]["arguments"]) for t in tc]}
# 2 tool with multiple args + follow-up turn with result
msgs = [{"role": "user", "content": "Find me flights from Boston to Denver on 2026-10-14."}]
m = chat(msgs, tools=[W, S]); tc = m.get("tool_calls") or []
a = json.loads(tc[0]["function"]["arguments"]) if tc else {}
ok2 = bool(tc) and tc[0]["function"]["name"] == "search_flights" and a.get("date") == "2026-10-14"
msgs += [{"role": "assistant", "content": m.get("content") or "", "tool_calls": tc},
         {"role": "tool", "tool_call_id": tc[0]["id"] if tc else "x", "content": json.dumps({"flights": [{"id": "UA123", "price": 212}, {"id": "B6 99", "price": 189}]})}]
m2 = chat(msgs, tools=[W, S])
ok3 = "189" in (m2.get("content") or "") or "B6" in (m2.get("content") or "")
res["tool_multi_turn"] = {"ok": ok2 and ok3, "args": a, "final": (m2.get("content") or "")[:200]}
# 3 strict JSON
m = chat([{"role": "user", "content": "Return ONLY a JSON object with keys name (string), age (integer), tags (array of 3 strings) describing a fictional astronaut. No prose."}], max_tokens=300)
c = re.sub(r"^```(json)?|```$", "", (m.get("content") or "").strip()).strip()
try:
    j = json.loads(c); okj = isinstance(j.get("age"), int) and len(j.get("tags", [])) == 3 and isinstance(j.get("name"), str)
except Exception: okj = False
res["json_strict"] = {"ok": okj, "out": c[:200]}
m = chat([{"role": "user", "content": "Give 4 planets."}], max_tokens=300, response_format={"type": "json_schema", "json_schema": {"name": "p", "schema": {"type": "object", "properties": {"planets": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4}}, "required": ["planets"]}}})
try: okg = len(json.loads(m["content"])["planets"]) == 4
except Exception: okg = False
res["json_guided"] = {"ok": okg, "out": (m.get("content") or "")[:200]}
# 4 reasoning (thinking on)
Q = [("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How many cents does the ball cost? Answer with just the number at the end as 'ANSWER: n'.", "5"),
     ("What is 17 * 23 + 144 / 12? End with 'ANSWER: n'.", "403"),
     ("If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? End with 'ANSWER: yes' or 'ANSWER: no'.", "yes"),
     ("How many times does the letter r appear in 'strawberry raspberry'? End with 'ANSWER: n'.", "6")]
rr = []
for q, a in Q:
    m = chat([{"role": "user", "content": q}], think=True, max_tokens=4000)
    got = re.findall(r"ANSWER:\s*\**\s*([\w.]+)", m.get("content") or "")
    rr.append({"ok": bool(got) and got[-1].lower().rstrip(".") == a, "got": got[-1:] , "reasoning_len": len(m.get("reasoning_content") or m.get("reasoning") or "")})
res["reasoning"] = {"ok": sum(x["ok"] for x in rr), "of": len(rr), "items": rr}
# 5 code: write function, execute tests
m = chat([{"role": "user", "content": "Write a Python function `merge_intervals(intervals)` that merges overlapping closed intervals given as a list of [start, end] and returns them sorted. Only output one python code block."}], max_tokens=800)
code = re.findall(r"```(?:python)?\n(.*?)```", m.get("content") or "", re.S)
test = "\nassert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]])==[[1,5]]\nassert merge_intervals([])==[]\nprint('PASS')"
try: okc = subprocess.run(["python3", "-c", (code[0] if code else "") + test], capture_output=True, text=True, timeout=10).stdout.strip() == "PASS"
except Exception: okc = False
res["code_exec"] = {"ok": okc}
res["summary"] = {k: (v["ok"] if k != "reasoning" else f'{v["ok"]}/{v["of"]}') for k, v in res.items()}
os.makedirs("results", exist_ok=True)
json.dump(res, open(f"results/gates_{sys.argv[1] if len(sys.argv) > 1 else 'run'}.json", "w"), indent=1)
print(json.dumps(res["summary"]))
