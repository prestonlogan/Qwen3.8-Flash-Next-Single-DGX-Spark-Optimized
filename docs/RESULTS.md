# Results

All measurements on one DGX Spark (GB10) with the pinned image and checkpoint. Method: [BENCHMARKING.md](BENCHMARKING.md).

## Common settings

Unless a row says otherwise:

| Setting | Value |
|---|---|
| Client | `bench/q38bench.py`, OpenAI streaming chat API |
| Max tokens | 800, `ignore_eos` (every request generates exactly 800 tokens) |
| Thinking | off (`chat_template_kwargs.enable_thinking=false`) |
| Streams | S=1 (one request at a time) |
| Greedy | temperature 0, top-p 1 |
| Sampled | stated per row; "default chat" = the model's `generation_config`: T1.0 / top-p 0.95 / top-k 20 |
| tok/s | per-stream decode rate (completion_tokens − 1) / (t_last − t_first), averaged over requests |
| ms/step, tok/step | engine `/metrics` deltas over the same requests |

Suites (prompts are in `bench/q38bench.py` and `bench/prose3_confirm.json`):

| Suite | Prompts | Role |
|---|---|---|
| anchor | PROSE[0], the sparkDash hash-map explanation prompt | technical prose, ≈3.0 tok/step; comparison with sparkDash only |
| PROSE ordinary-5 | PROSE[1..5]: story, history essay, letter, immune-system explainer, travel guide | historical speed gate (E01–E39) |
| PROSE | all 6 PROSE prompts | used in the later same-process A/Bs |
| PROSE2 | 12 prompts, disjoint from PROSE and training data | sealed confirmation for T8; later dev/regression set |
| PROSE3 | 8 prompts, `bench/prose3_confirm.json` (sha256 `7319b0b1…`) | fresh confirmation set for E42 |
| code / json / structured | 2 / 1 / 1 prompts | high-acceptance workloads; never compare with prose |

## Measurement types

| Type | Meaning |
|---|---|
| **direct matched A/B** | both arms in one loaded process, interleaved (off/on/on/off), same prompts and settings |
| **promoted confirmed** | measured on the promoted stack as a confirmation/control run |
| **sequential** | stack measured on its own load/launch; compared with an earlier run (launch-to-launch drift included) |
| **pooled** | mean of the arms of several alternating A/Bs on the same stack |
| **chained estimate** | product of separately measured A/Bs, not a single measurement |
| **single-run preliminary** | one pass, no control |

## 1. Baseline: the Mia recipe as deployed (E01)

Isolated launch with all production arguments (before the PLE row fix). 2 repetitions per prompt.

| Suite | S | Sampling | tok/s (per stream) | Aggregate | ms/step | tok/step |
|---|---|---|---|---|---|---|
| anchor (hash-map) | 1 | greedy | 49.62 | | 60.31 | 2.996 |
| **PROSE ordinary-5** | 1 | greedy | **36.66** | | 59.46 | 2.183 |
| PROSE 6-prompt mean | 1 | greedy | 38.8 | | 59.6 | 2.32 |
| PROSE ordinary-5 | 1 | T0.7 / top-k 20 | 34.85 | | 61.10 | 2.131 |
| code (2 prompts) | 1 | greedy | 55.8 / 57.7 | | 60.2 | 3.42 |
| json | 1 | greedy | 53.5 | | 61.5 | 3.30 |
| structured (count to 200) | 1 | greedy | 62.7 | | 61.6 | 3.87 |
| PROSE 6 prompts | 2 | greedy | 30.8–34.4 | 58.4–60.1 | 73.8 | 2.26 |

KV capacity 1,130,587 tokens (4.31× a 262k request); load 392 s. Mia's published "48.7 tok/s prose" is the sparkDash
hash-map prompt on Mia's own profile; it corresponds to our anchor row, not to ordinary prose.

## 2. Progression (greedy, S=1, PROSE ordinary-5)

Each row adds one item. "vs E01" uses 36.66. Rows before E11 were measured on the PLE-bug model (the fix is speed-neutral).
Absolute tok/s drifts between launches and sessions by ~±2%; ms/step is the more robust column.

| Stack | Added | ordinary tok/s | ms/step | vs E01 | Type |
|---|---|---|---|---|---|
| E01 | Mia recipe baseline | 36.66 | 59.46 | — | baseline |
| E04 | FP8 draft head + 64,877-token draft vocab | 39.45 | 58.03 | +7.6% | sequential |
| E11 | PLE row fix (correctness) | 39.44 | 57.8 | +7.6% | sequential |
| E12 | INT4 coarse + exact refine target head | 41.84 | 55.09 | +14.1% | sequential |
| E13 | W8 HC mixers | 44.11 | 52.0 | +20.3% | sequential |
| E18 | draft dense W8 | 44.41 | 51.45 | +21% | sequential |
| E20 | async PLE handshake | 44.78 | 51.52 | +22% | sequential |
| E22/E22d | W6A16 dense projections (MXFP8 freed) | 46.3 (pooled 46.69 / 45.96) | 49.27–49.64 | +26% | sequential |
| E26 | draft graph padding removal | ≈47.45 | 48.39–48.64 | +29% | sequential |
| E30 | MoE tactic (bitwise identical) | 48.08 | 47.82 | +31% | sequential |
| E32 | draft head INT4 + FP32 refine | 48.34 | 47.46 | +32% | A/B arm |
| E33 | target head INT2 coarse | 48.88 | 47.01 | +33% | A/B arm |
| E34 | draft head INT2 + refine top-64 | 49.03 | 46.53 | +34% | A/B arm |
| E35 | PLE zero-copy rows | 49.89 | 45.43 | +36% | A/B arm |
| E36 | PLE shared-memory request slot | 51.14 | 44.93 | +39% | A/B arm |
| E37 | Triton BF16 router / in_proj_ba GEMVs | 51.65 | 44.80 | +41% | A/B arm |
| E38 | fused HC split-K reduce | 51.38 (pooled 51.2–51.7) | 44.62 | ≈+40% | A/B arm / pooled |
| E39 | fast-head exactness gate (C=512) | greedy unchanged | 43.2–43.8 | | gate check |

The E38 milestone pass (single run) drew low-acceptance samples: ordinary 49.46 tok/s at 44.6–45.7 ms/step. The pooled
A/B means are the better estimate.

## 3. Late stack (E40–E42), direct matched A/Bs

From E40 on, results are reported per same-process A/B on the full 6-prompt PROSE set or on PROSE2/PROSE3 rather than as
ordinary-5 means.

### E40 — r32a drafter (T8), off → on in one process

| Workload | Sampling | n/arm | tok/s off → on | tok/s Δ | tok/step Δ |
|---|---|---|---|---|---|
| PROSE (6) | greedy | 12 | 53.65 → **55.59** | +3.6% | +4.7% (2.405 → 2.519) |
| PROSE2 (then sealed) | greedy | 12 | 52.15 → 53.92 | +3.4% | +4.3% |
| PROSE | T0.7 / p0.95 / k20 | 12 | 46.17 → 48.53 | +5.1% | +6.1% |
| PROSE2 | T1.0 / p0.95 / k20 | 12 | 42.36 → 43.50 | +2.7% | +3.1% |
| code | greedy | 4 | 74.97 → 76.92 | +2.6% | +4.4% |
| JSON | greedy | 2 | 75.13 → 80.35 | +6.9% | +6.5% |
| code | T0.7 | 2 | 63.76 → 67.02 | +5.1% | +6.0% |
| JSON | T0.7 | 1 | 66.93 → 67.25 | +0.5% (n=1) | −1.2% |
| prose at 150k context | greedy | 3 | 49.62 → 51.34 | +3.5% | +3.5% |
| prose at 150k context | T0.7 | 3 | 41.48 → 42.76 | +3.1% | +3.2% |
| PROSE, S=2 aggregate | greedy | 6 waves | 75.72 → 78.82 | +4.1% | +5.1% |

A two-launch comparison gave +6.0% tok/s (54.04 → 57.27); the same-process +3.6% is the reliable figure.

### E41 — probabilistic draft for sampled requests, off (greedy draft) → on

| Workload | Sampling | n/arm | tok/s off → on | Δ | Type |
|---|---|---|---|---|---|
| PROSE2 | T0.7 / p0.95 / k20 (top-20 draft, S2) | 24 | 46.36 → 48.11 | +3.8% | direct A/B |
| PROSE2 | T1.0 / p0.95 / k20 (top-20 draft, S2) | 12 | 42.68 → 47.29 | +10.8% | direct A/B |
| PROSE2 | T1.0 / p0.95 / k20, draft shaping `0.9 0.9` vs plain top-20 (S3) | 24 | | +3.56% | direct A/B |
| PROSE2 | T1.0 / p0.95 / k20, S2 + S3 | | 42.68 → ≈49.0 | ≈+14.7% | **chained estimate** |
| PROSE | T0.7 | 12 | 48.15 → 50.56 | +5.0% | direct A/B |
| PROSE, S=2 aggregate | T0.7 | | 37.01 → 39.67 | +7.2% | direct A/B |
| code + JSON | T0.7 / T1.0 | 6 / 3 | 69.34 → 69.96 / 67.47 → 67.36 | n.s. | direct A/B |
| PROSE2 | greedy (must be unaffected) | | 54.37 → 54.37 | 0.0% | direct A/B |
| 150k context | T0.7 / T1.0 | 2 | 45.08 → 46.03 / 43.09 → 45.19 | +2.1% / +4.9% | direct A/B |

Direct sampled measurements of the E41 stack on PROSE2 at T1.0/p0.95/k20: **48.25 tok/s** (S3 arm, n=24) and **48.38 tok/s**,
49.02 ms/step (S6 control arm, n=12). The "≈49 tok/s sampled" figure was the chained estimate.

### E42 — top-k/top-p fast path, E41 → E42 (U30a v2)

| Set | Sampling | n/arm | ms/step off → on | Paired Δ ms/step | Faster prompts | tok/s off → on |
|---|---|---|---|---|---|---|
| PROSE2 (dev) | T1.0 / p0.95 / k20 | 24 | 49.92 → **48.96** | −0.96 ± 0.13 | 12/12 | 48.48 → **49.07** |
| PROSE3 (fresh confirmation) | T0.7 / p0.95 / k20 | 16 | 49.73 → **48.82** | −0.91 ± 0.08 | 8/8 | 47.39 → **48.86** |

The tok/s deltas include acceptance noise (PROSE3 tok/step differed by +1.2% between arms); the ms/step delta is the
reliable effect (≈+1.9% at constant tok/step). Greedy requests are not affected by E42.

### Current promoted numbers (E42)

| Workload | Sampling | tok/s | Type |
|---|---|---|---|
| PROSE (6 prompts) | greedy | 55.59 | direct A/B arm (E40; greedy path unchanged by E41/E42) |
| PROSE2 | greedy | 54.37 | promoted confirmed (E41 A/B, greedy arm) |
| PROSE2 | T1.0 / p0.95 / k20 | 49.07 (48.96 ms/step) | direct A/B arm |
| PROSE3 | T0.7 / p0.95 / k20 | 48.86 (48.82 ms/step) | direct A/B arm |
| PROSE2 (clean checkout via `./run.sh`) | greedy | 53.86 (2.363 tok/step, 43.80 ms/step), n=12 | clean-copy smoke, single pass, not an A/B |
| PROSE3 (clean checkout via `./run.sh`) | T1.0 / p0.95 / k20 | 48.15 (2.347 tok/step, 48.66 ms/step), n=8 | clean-copy smoke, single pass, not an A/B |
| anchor | greedy | 68.85 (3.015 tok/step) | single-run preliminary (E40 launch) |

## 4. Step anatomy

| ms | E00 (Mia, profile) | E26 era | E41 greedy | E41 sampled T1.0/p0.95/k20 |
|---|---|---|---|---|
| Whole step | 62.35 (median) | 45.0 (exec mean) | 42.2 | 47.5 |
| Verify graph | 43.5 | 40.7 (median) | 38.1 | 38.5 |
| Target head + sample | 5.3 (head) | 1.13 (host sample) | 1.36 | 6.43 |
| Propose | ≈8.5 | 4.37 | 3.43 | 3.45 |
| Idle | ≈5 | 0.54 | 0.52 | 0.55 |

E42 removes ≈0.9 ms from the sampled head+sample phase (section 3).

## 5. Long context and concurrency

| Check | Stack | Result |
|---|---|---|
| Needle retrieval | E22 | correct at 108k and 201,650 prompt tokens (201k prefill took 96 s) |
| Decode vs context (greedy, same prompt tail) | E40/E41 (L1) | 54.84 / 53.05 / 52.64 / 52.36 / **52.74** tok/s at 351 / 1,555 / 8,051 / 40,055 / 150,053 prompt tokens; tok/step 2.31–2.39 (flat) |
| Two concurrent long conversations | E38 | 144,047-token needle correct while a 150,053-token conversation decodes at 47.6 tok/s; min MemAvailable 12.9 GB |
| Same | E39 (V39) | needle OK; concurrent decode 50.3 tok/s; min MemAvailable 13.0 GB |
| Same | E40 (T8 check 3) | needle OK; concurrent decode 48.82 tok/s; min MemAvailable 13.5 GB |
| Same | E41 (S4) | needle OK; warm matched decode 50.87 (off) vs 50.25 (on) tok/s; min MemAvailable 12.8 GB |
| KV capacity | E36/E38 | 1,115,942 tokens = 4.26× a 262k request |
| S=2 aggregate, PROSE greedy | E40 | 78.82 tok/s |
| S=2 aggregate, PROSE T0.7 | E41 | 39.67 tok/s |

In the first S4 dual run both 144k/150k prefills were cold and overlapped the decode (5.62 tok/s for the decode stream);
the matched warm rerun is the representative number. The memwatch floor is 6 GB; it never triggered in these runs.

## 6. Quality gates

| Gate | Method | Result on the stack |
|---|---|---|
| Perplexity | teacher-forced, 6 human-written Wikipedia extracts, 7,601 tokens, via `prompt_logprobs` | reference band 2.383–2.400 (BF16/MXFP8, fixed PLE); every retained step 2.367–2.398 (e.g. E22 2.374–2.394, E33 2.391, E36 2.371, E38 2.378, V39 2.390). PLE-bug baseline: 4.998. |
| Top-1 agreement | teacher-forced, vs a BF16 reference run | identical-run floor 93.66%; W6 93.91%; E20 stack 93.76% |
| Tool calling | `bench/gates.py`: single call; multi-argument call + follow-up turn with the tool result | pass (every milestone; E41 with all items on; clean-checkout E42) |
| JSON | strict JSON and schema-guided JSON | pass |
| Reasoning | 4 problems with thinking on (`qwen3` parser) | 4/4 |
| Code | generated code executed against tests | pass |
| Sampled distribution | mean target log-prob of sampled texts, rescored with the exact head (`bench/rescore.py`) | E41 on − off: T1.0 +0.054 ± 0.104 SE, T0.7 +0.018 ± 0.032 SE (noise) |
| Exactness counters | in-server comparison of fast vs reference paths | target head: 0 argmax differences at C=512 on 3,792 sampled-context rows; draft head K=64: 0 non-tie flips; PLE: 0 row mismatches over 1,744 steps; top-k/top-p: 0 of 2,048 rows differ |
