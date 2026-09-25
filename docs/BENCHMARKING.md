# Benchmarking

How the numbers in this repository were measured, and how to reproduce or extend them without fooling yourself.

## Metrics

`bench/q38bench.py` streams chat completions against the local server. Every request uses:

- thinking off (`chat_template_kwargs.enable_thinking=false`);
- `ignore_eos`, so every request generates exactly `--max-tokens` (default 800);
- a fixed seed.

It reports:

| Metric | Definition | Use |
|---|---|---|
| **decode tok/s** | `(completion_tokens − 1) / (t_last_token − t_first_token)`, the sparkDash DecodeBench definition | headline speed (excludes prefill/TTFT) |
| **ms/step** | engine step time from `/metrics` deltas over the request | cost of one verify + draft step; **the robust metric for kernel/step-cost changes** |
| **tok/step** | committed tokens per engine step (1 + accepted drafts), from `/metrics` spec-decode counters | draft acceptance; **noisy per prompt and per sampled run** |
| per-position acceptance | acceptance rate of draft positions 1/2/3 | where the drafter loses |

`tok/s ≈ 1000 × tok/step / ms/step`. A change that only touches step cost should be judged on ms/step. Its tok/s delta also carries acceptance noise; for sampled requests that noise is several percent per prompt.

## Suites

| Suite | Content | Role |
|---|---|---|
| `anchor` | the sparkDash "hash-map explanation" prompt (PROSE[0]) | continuity with the sparkDash figure. **Not ordinary prose**: ≈3 tok/step. |
| `prose` | 6 prompts (anchor + 5 ordinary: story, history essay, letter, immune system explainer, travel guide) | historical development set. "Ordinary-5" = prompts 1–5. |
| `prose2` | 12 ordinary prose prompts in genres and topics disjoint from the drafter training corpus | written as a sealed confirmation set. It has since been used for selection, so it is now the **dev/regression** set. |
| `bench/prose3_confirm.json` | 8 fresh ordinary prose prompts (sha256 `7319b0b1…`), grep-checked disjoint from the training corpora | **confirmation set**. Use it only when a candidate is ready for promotion. |
| `code`, `json` | 2 code prompts, 1 JSON-generation prompt | high-acceptance structured workloads |
| `structured` | "count from 1 to 200" | upper bound on acceptance; never report as prose |

`--suite` also accepts a path to a JSON list of prompts.

## Greedy vs sampled

The model's `generation_config` is **T 1.0 / top-p 0.95 / top-k 20**, which is what a chat client gets by default. Greedy (T=0) and sampled requests take different code paths in this stack:

- The fast target head is used only for greedy batches.
- Sampled requests use the exact full-vocabulary head, the probabilistic draft and the top-k/top-p fast path.

Always report greedy and sampled separately:

```bash
python3 bench/q38bench.py --tag G --suite prose2 --out results/g.jsonl                        # greedy
python3 bench/q38bench.py --tag S --suite prose2 --temperature 1.0 --top-p 0.95 --top-k 20 \
        --out results/s.jsonl                                                                  # default chat sampling
python3 bench/t8_summ.py results/s.jsonl                                                       # per-tag means + paired comparison
```

## Same-process A/B (the protocol behind every "direct matched" number)

Step time varies by about ±1 ms from one server launch to the next (CUDA graph placement, autotune choices). Comparing two launches can therefore invent or hide a 2% effect.

Every retained optimization is a live toggle, so A/Bs run inside **one** loaded server:

```bash
bench/toggle_ab.sh results/myab exec_topkp.py topkp.txt prose2 2 --temperature 1.0 --top-p 0.95 --top-k 20
# arms alternate off/on/on/off per block; ACMD/BCMD override the control strings (default "off"/"on");
# the script ends by restoring FINAL (default = BCMD) so the promoted setting is left active.
python3 bench/t8_summ.py results/myab.jsonl
```

Analyse the results per prompt, pairing the on and off arms of the same prompt:

- report the mean and SE of the per-prompt differences, and how many prompts improved;
- for step-cost changes, report the paired ms/step delta.

## Long runs and tracked jobs

SSH sessions to the Spark can drop, so long runs go through `bench/xjob.sh`. It is a small job tracker:

- it records the PID and `/proc` start time (identity), the exit code, and a log;
- `wait` is bounded and prints a process-tree report on timeout;
- `kill` signals only that job's process group.

```bash
bench/xjob.sh start ab1 bash bench/toggle_ab.sh results/ab1 exec_topkp.py topkp.txt prose2 2
bench/xjob.sh wait ab1 1800      # exit 124 on timeout (job left running), otherwise the job's rc
```

## Other tools

- `bench/gates.py [TAG]` runs the functional gates: tool calling (single, multi-turn), strict and schema-guided JSON, reasoning 4/4, code execution. It writes `results/gates_TAG.json`. Env: `MODEL`, `Q38_PORT`.
- `bench/t8_long.py N_words [T] [tag]` runs a long-context decode (filler + question) and prints tok/s, tok/step and acceptance.
- `bench/rescore.py` rescores saved sampled texts under the target model. It is a sanity check that a draft-side change did not move the output distribution.

## Pitfalls we hit

- **Workloads are not comparable.** Code, JSON, counting and the anchor prompt accept far more drafts than ordinary prose.
- **Dev-set reuse.** Any set used to choose between variants stops being a confirmation set. Keep a fresh disjoint set for promotion decisions and do not regenerate it after every edit.
- **Chained estimates are not measurements.** A→B and B→C A/Bs multiplied together are an estimate of A→C. Label them.
- **Offline wins shrink when served.** Offline or teacher-forced gains under ≈5% have not survived served A/Bs. Gate engineering work on a ≥5% offline estimate.
- **Cold prefix cache.** Two concurrent 150k-token prefills make a decode look 10× slower. Warm the caches or match conditions.

## SSD hygiene (experiment-local; no host or Docker-daemon changes)

- **Container log:** `serve.sh` passes `--log-driver json-file --log-opt max-size=50m --log-opt max-file=3` for this container only, so at most ≈150 MB. `serve.sh` also saves a copy of the log to `.state/logs/<tag>-*-container.log` at startup.
- **JIT caches:** Triton (`/root/.triton`) and CUDA (`/root/.nv`) are bind-mounted from `.state/cache/jit/<image-digest-12>/{triton,nv}`. That's about 0.3 GB, reused across relaunches instead of being rewritten each launch.
  - Keying by image digest isolates toolkit and compiler changes. Triton and CUDA also hash kernel source and options internally, so edited kernels recompile.
  - These caches are never shared with production.
- **Short-lived large data** (hidden-state captures, training tensors) goes to `/dev/shm` only while free RAM allows. Delete it when done.
- **Safe cleanup** (experiment-owned only):
  - Stop the server first, then run `rm -rf .state/cache/jit/<old-digest>` for digests of images no longer used.
  - Old `.state/logs/*` can be deleted freely.
  - Never delete the Hugging Face cache, the packed PLE table, or other Docker objects.
