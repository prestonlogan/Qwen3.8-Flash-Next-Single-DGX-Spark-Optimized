# Qwen3.8-Flash-Next NVFP4 on ONE DGX Spark — decode-optimized

> **Built on [MiaAI-Lab's single-DGX-Spark recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
> and their [`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4) checkpoint.**
> The launch recipe, the patched vLLM model files, the packed-PLE builder and the memory watchdog in this repository come
> from MiaAI Lab's work (upstream commit `d038090`). This repository adds a decode-speed optimization stack on top of it.
> See [Credits](#credits).

This repository serves the **original** `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` checkpoint on a **single NVIDIA DGX Spark**
(GB10, SM121, 128 GB unified memory) with vLLM, then hot-installs a stack of decode optimizations into the running server.
It is **not** the 5-expert INT4 "FAST" variant or any other smaller or pruned model: every routed expert, the full top-k
routing and the full vocabulary are kept. The checkpoint weights are untouched. The only learned change is a small LoRA
on the MTP *drafter*, and the drafter affects speed only: speculative decoding verifies every token with the target.

Every retained optimization falls into one of three correctness classes:

- **output-exact**: bit-identical greedy output, or distribution-exact sampling via rejection sampling;
- **high-fidelity**: weights stored losslessly, but logits not bit-identical (HX2);
- **quality-gated numeric change**: W6/W8 dense projections, gated on perplexity and task checks.

Each item says which class it belongs to in [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md). The current stable, default
configuration is **E46**:

- the E44 base (r32a drafter, fast heads, PLE zero-copy, the prefix-cache block-retention backport);
- plus **QF1**, a faster NVFP4 decode MoE kernel adapted from SSHdotCodes' Apache-2.0 qwenfast kernels;
- plus **HX2**, a high-fidelity compressed BF16 head for sampled requests.

**Project status (2026-09-25): paused.** E46 is the frozen public recipe. The research log lists what was tried,
including what did not work.

| | |
|---|---|
| Hardware | 1× DGX Spark (GB10, SM121, 128 GB unified LPDDR5X) |
| Checkpoint | `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` @ `925d7be6c14c6c9442ef83e8f05b5a3c39304f69` (~100 GB, not included) |
| Image | `vllm/vllm-openai@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8` |
| Engine | day-0 Qwen3.8 vLLM fork build `0.1.dev20073+g8e685d198`, FlashInfer 0.6.17, torch 2.13.0+cu130 |
| Serving profile | TP=1, 262,144 context, FP8 KV cache, MTP speculative decoding (3 draft tokens), `max-num-seqs` 4 |
| Stable stack (default) | **E46** (E45 + HX2 high-fidelity compressed head for sampled requests: weights stored losslessly, logits not bit-identical) — see [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) |
| Optional (off by default) | **S6FX** sampled fast head (`Q38_S6=1`): high-fidelity, not exact — see below |

## Why this exists

Single-stream decode on one Spark is memory-bandwidth-bound. The Mia recipe makes the 99 GB checkpoint fit and serve
correctly on one box; this work asks how much faster single-stream (and two-stream) decode can be made **without
changing what the model outputs**. Along the way it found and fixed a correctness bug in the PLE CPU-offload path of the
recipe version we started from (see section B1 of [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md)).

## Headline results

All numbers are decode tokens/s (not prefill/TTFT) for one stream (S=1) unless stated, 400–800 generated tokens, thinking off, `ignore_eos`, on one DGX Spark.
"Ordinary prose" means unpredictable natural-language prose prompts (stories, essays, letters, explainers) and is the
hard case for speculative decoding. Each number is labelled with how it was measured; see
[docs/RESULTS.md](docs/RESULTS.md) for the full tables and [docs/BENCHMARKING.md](docs/BENCHMARKING.md) for the method.

| Workload | Sampling | Result | Measurement type |
|---|---|---|---|
| **Ordinary prose — PROSE2 (12 prompts), E46** | greedy | **56.83 tok/s** (42.22 ms/step, 2.405 tok/step) | **current stable stack, direct** (fresh boot, 400 tokens, R36 control arm) |
| **Ordinary prose — PROSE2, E46** | default chat T1.0 / top-p 0.95 / top-k 20 | **52.02 tok/s** (45.52 ms/step, 2.373 tok/step) | current stable stack, direct (same run) |
| Ordinary prose — PROSE3 (8 fresh prompts), E46 | greedy | 54.67 tok/s (42.12 ms/step, 2.306 tok/step) | current stable stack, direct (same run) |
| Code (2 prompts) / JSON (1 prompt), E46 | greedy | 81.6 / 84.7 tok/s (3.47 / 3.71 tok/step) | current stable stack, direct, n=4 / n=2, 400 tokens (final check) |
| Code / JSON, E46 | default chat T1.0 | 74.8 / 71.2 tok/s | current stable stack, direct, n=4 / n=2 (final check) |
| Two long conversations (150k + 150k context), E46 | greedy | ≈40.2 tok/s **per stream** (≈80.5 aggregate); min MemAvailable 13.3 GB | current stable stack, direct (warmed dual run) |
| *Rows below are historical (earlier stacks) or baselines, kept for provenance.* | | | |
| Ordinary prose, **baseline** (Mia recipe as deployed, E01) — PROSE ordinary-5 | greedy | **36.66 tok/s**, 59.5 ms/step, 2.18 tok/step | isolated baseline run |
| Ordinary prose, **baseline** (E01) — PROSE ordinary-5 | T0.7 / top-k 20 | 34.85 tok/s | isolated baseline run |
| Ordinary prose, E38 stack (before drafter adapter) — same PROSE ordinary-5 set | greedy | 51.2–51.7 tok/s (**≈+40% vs E01**), ≈44.6 ms/step | pooled alternating A/B arms |
| Ordinary prose, promoted drafter (E40) — PROSE, 6 prompts | greedy | 53.65 → **55.59 tok/s** (+3.6%) | direct matched A/B (same process), historical |
| Ordinary prose — **PROSE ordinary-5** (same prompts as E01) | greedy | **53.23 tok/s** (2.387 tok/step, 44.80 ms/step), n=10 → **+45.2% vs E01** | E42 stack, direct (CK42), historical — E45/E46 add ≈+4% greedy on PROSE3 in matched A/B |
| Ordinary prose — PROSE2 (12 prompts, dev set) | greedy | E42 stack: base drafter 51.19 → **r32a 53.73 tok/s** (2.399 tok/step, 44.61 ms/step), +4.99% ± 0.61, 12/12 | direct matched A/B, E42 stack (CK42), historical |
| Ordinary prose — PROSE2 | **default chat** T1.0 / top-p 0.95 / top-k 20, thinking off | E40 state 43.69 → **E42 48.57 tok/s** (2.387 tok/step, 49.11 ms/step), +11.2% ± 1.2, 12/12 | direct matched A/B (CK42) |
| Ordinary prose — PROSE3 (fresh confirmation set) | default chat T1.0 / p0.95 / k20 | E40 state 42.48 → **E42 47.83 tok/s**, +12.7% ± 1.1, 8/8 | direct matched A/B (CK42) |
| Same, E43 (exactness fix) vs E42 | default chat | PROSE2 48.39 → 48.86 (+0.9% ± 1.0); PROSE3 47.60 → 47.77 (+0.4% ± 0.4): speed-neutral | direct matched A/B (CK1) |
| Code (2 prompts) / JSON (1 prompt) | greedy | 78.00 / 79.21 tok/s (n=4 / n=2) | E42 stack, direct (CK42), historical |
| Code / JSON | default chat T1.0 | 68.34 / 71.22 tok/s (n=4 / n=2) | E42 stack, direct (CK42), historical |
| Ordinary prose, 2 concurrent streams (S=2) | greedy / T1.0 | 78.68 / 72.78 tok/s aggregate (41.9 / 38.9 per stream) | E42 stack, direct (CK42), historical |
| 150k-token context decode | greedy / T1.0 | 46.8–50.9 / 47.0–47.6 tok/s | E42 stack, direct (CK42, 2 runs each), historical |
| **Warm multi-turn TTFT** (responsiveness, not decode), ~8.4k-token conversation | greedy | second turn 1.24 → **0.68 s**; after a ~3k-word tool result 3.51 → **2.65 s**; cold first turn unchanged (5.2 s) | direct matched A/B, E43 vs E44 (6 paired sessions) |
| **E46 vs E45** (compressed high-fidelity head) | default chat T1.0 | ms/step −1.42 ± 0.22 (PROSE2, 12/12), −1.58 ± 0.37 (PROSE3, 8/8) ≈ −3%; greedy unchanged; not bit-identical: 99.98% of logits bitwise-equal, argmax equal, TV ≤ 2.4e-6 (accumulation order) | direct matched A/B, same process (HX2) |
| **E45 vs E44** (NVFP4 decode MoE kernel) | greedy / default chat T1.0 | PROSE3 greedy 53.08 → **55.17** (+4.0% ± 0.6, 8/8); PROSE3 T1.0 48.21 → **49.50** (+2.8% ± 1.3); PROSE2 T1.0 49.01 → **50.72** (+3.6% ± 1.0); 150k greedy ≈+6–8% preliminary (n=2 per arm, cold/warm imbalance); tok/step unchanged | direct matched A/B, same process (QF1/QF3) |
| Clean-checkout smoke of E42 via `./run.sh` | PROSE2 greedy / PROSE3 T1.0 | 53.86 / 48.15 tok/s | clean-copy smoke, single pass, not an A/B |

Notes that matter when reading the table:

- **Workloads are not interchangeable.** Code, JSON and "technical" prose (such as the hash-map explanation prompt used by
  sparkDash) accept far more draft tokens per step than ordinary prose, so their tok/s is much higher at the same step
  time. Never compare a code/JSON/anchor number with an ordinary-prose number.
- **Sampled is slower than greedy** at the same prompts: the exact full-vocabulary target head is used for sampled requests
  (≈+5 ms/step), and sampled drafts are accepted less often.
- The "≈49 tok/s sampled" figure quoted during E41 development was a **chained estimate**; the direct measurements are the
  ones in the table.
- PROSE2 and PROSE3 were not run on the E01 baseline, so there is no matched baseline number for them.

### Comparison with the original Mia recipe

| Number | What it measures | Where it comes from |
|---|---|---|
| **48.7 tok/s** | The single-stream sparkDash "prose" decode figure that was our starting reference. The sparkDash prose prompt is the *hash-map explanation* prompt (≈3.0 tok/step, a highly predictable technical text). | production reference at project start |
| **49.62 tok/s** | The same hash-map prompt ("anchor") on our isolated, production-identical launch of the Mia recipe (E01). Consistent with 48.7. | E01 |
| **36.66 tok/s** | The five *ordinary* prose prompts on that same E01 launch (≈2.18 tok/step, same ≈59.5 ms/step). | E01 |

Our baseline for all "vs E01" statements is the 36.66 ordinary-prose number, measured on the same prompt set as the
later stack. The 48.7 figure belongs to a different workload (the anchor prompt) and is not used as a baseline for
ordinary prose. On the anchor prompt the stack reaches ≈61–69 tok/s depending on configuration (e.g. 68.85 tok/s with
the E40 drafter), but that is not a headline metric here.

## Quick start

Prerequisites: one DGX Spark with Docker and the NVIDIA container runtime; `python3`, `curl`, `ss`; ≥110 GiB free disk
for the checkpoint plus ~27 GiB for the packed PLE table; **≥104 GiB `MemAvailable`** (stop other model servers first).

```bash
cp .env.sample .env          # optional; every variable has a default
./download.sh                # EXPLICIT ~100 GB download of the pinned checkpoint (asks to confirm; --yes to skip)
./run.sh --dry-run           # run every host check and print what would happen; builds and launches nothing
./run.sh                     # checks → prepare (PLE table, drafter patch) → launch → install E46 stack → health
./stop.sh                    # stop the container and its memory watchdog (keeps a bounded log tail)
```

Nothing except `./download.sh` downloads the model. When `./run.sh` finishes, an OpenAI-compatible API is at
`http://127.0.0.1:5810/v1` with model name `qwen3.8-flash-next`. Loading takes several minutes (E01: 392 s to `/health`);
the optimization install afterwards takes ~10–20 s.

### Modular path

| Script | What it does |
|---|---|
| `download.sh [--yes]` | Download the pinned checkpoint revision into the HF cache using the image's `huggingface_hub` (resumable); records `refs/main` if missing. |
| `scripts/prepare.sh` | One-time host prep, no download: builds the packed PLE table (~27 GiB, CPU-only, Mia's builder), compiles the one-line ARM barrier helper `overlays/runtime/exp_dmb.c`, and builds the drafter patch. |
| `scripts/build_adapter.sh` | Merges the committed r32a LoRA (14.6 MB) into the checkpoint's BF16 MTP tensors → `adapters/r32a/mtp_patch_r32a.safetensors` (175 MB), network-less and CPU-only, and verifies SHA-256 `bd3c1807…` (bit-identical to the patch used for all measurements). |
| `start.sh` → `scripts/serve.sh` | Launch the container with the boot-time overlays, wait for health, then run `scripts/install_stack.sh`. |
| `scripts/install_stack.sh` | Hot-install the stable E46 stack into a running server (see [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md)). |
| `stop.sh` → `scripts/stop.sh` | Stop the watchdog and the container. |

## Configuration

Every script sources `scripts/common.sh`, which loads `.env` (if present) and holds the single set of defaults, so the
same values apply to `./run.sh`, `./start.sh`, `./stop.sh` and the helper scripts. Environment variables set in the
shell also work. `serve.sh` passes `--revision` and `--tokenizer-revision` with the pinned commit, so the server
resolves exactly the downloaded snapshot while running offline (`HF_HUB_OFFLINE=1`).

| Variable | Default | Meaning |
|---|---|---|
| `Q38_PORT` | `5810` | API port |
| `Q38_NAME` | `q38fn-opt` | container name |
| `Q38_SERVED_NAME` | `qwen3.8-flash-next` | model name in the API |
| `Q38_HF_HOME` | `$HOME/.cache/huggingface` | host HF cache holding the checkpoint |
| `Q38_PLE_CACHE` | `$HOME/.cache/vllm/ple_cache` | root of the packed PLE table |
| `Q38_STATE` | `<repo>/.state` | logs and the vLLM compile/autotune cache |
| `Q38_GMU` | `0.786` | `--gpu-memory-utilization` |
| `Q38_MAX_NUM_SEQS` | `4` | `--max-num-seqs` |
| `Q38_MAX_MODEL_LEN` | `262144` | `--max-model-len` |
| `Q38_NO_ADAPTER` | `0` | `1` = serve the stock MTP drafter instead of the r32a patch |
| `Q38_NO_INSTALL` | `0` | `1` = launch only; do not hot-install the stack (serve.sh) |
| `Q38_IMAGE`, `Q38_MODEL_REVISION` | pinned values | overrides; untested with anything but the pins |
| `HF_TOKEN` | unset | only if your HF setup needs it; the checkpoint is public |

Fixed launch settings (in `scripts/serve.sh`): FP8 KV cache, BF16 Mamba/GDN state cache, chunked prefill with
2,048 batched tokens, `--speculative-config {"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}`,
CUDA graphs `FULL_DECODE_ONLY` at capture sizes 4/8/12/16, V2 model runner, `qwen3` reasoning parser, `qwen3_xml` tool
parser with auto tool choice, and the chat template shipped in Mia's recipe.

## Using the endpoint

```bash
curl -s http://127.0.0.1:5810/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role": "user", "content": "Write a haiku about unified memory."}],
  "max_tokens": 200,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

- **Sampling defaults.** Requests that omit sampling parameters use the checkpoint's `generation_config`: default chat
  T1.0, top-p 0.95, top-k 20. Speed tables label **greedy** (T=0) and **sampled** (default chat) separately. Sampled
  decode is slower because the verify step must produce the full target distribution.
- **Thinking.** It is on by default (`qwen3` reasoning parser). Pass `"chat_template_kwargs": {"enable_thinking": false}` to
  turn it off. Tool calling uses the `qwen3_xml` parser with `tool_choice: auto`.
- **Context and concurrency.** `max-model-len` is 262,144 and `max-num-seqs` is 4. The FP8 KV cache holds ≈1.1 M tokens,
  so two 150k–250k conversations run concurrently. Speed was tuned for 1–2 streams.
- **TTFT / prefix cache.** Automatic prefix caching is on. The E44 backport keeps the trailing cached block, so a warm
  follow-up turn re-prefills only the new text. These are TTFT numbers, not decode.

### Optional features and how to disable things

| Env var (for `./run.sh` / `./start.sh`) | Default | Effect |
|---|---|---|
| `Q38_S6=1` | off | **S6FX** sampled fast head: an INT4 top-1024 shortlist, then HX2-row refine, with full-head fallback outside the validated gate. On sampled prose it measured **+3.9% tok/s** (−1.9 ms/step, 12/12 prompts). **Caveat:** high-fidelity, *not exact*. It had 0 true top-k/top-p misses on 4,098 hard replay rows, and its final-distribution TV ≤1e-3 vs full BF16 is the same as E46, but it is not certified for every input. Leave it off if you need E46 fidelity. |
| `Q38_HX=0` | on | use the plain cuBLAS BF16 head for sampled requests (E45 behaviour, ≈3% slower sampled) |
| `Q38_QF=0` | on | use FlashInfer CUTLASS for the decode MoE (E44 behaviour) |
| `Q38_NO_ADAPTER=1` | off | serve the stock MTP drafter |
| `Q38_NO_INSTALL=1` | off | launch the stock recipe profile without the runtime stack |
| `Q38_BLOCK_DROP=1` | off | stock trailing prefix-cache block drop (disables the E44 backport) |
| `Q38_HOST` | `127.0.0.1` | bind address. The dev-mode RPC endpoint executes code, so never bind it to an untrusted network. |

### Benchmarking fairly and recovering

- Use `bench/q38bench.py` (see [docs/BENCHMARKING.md](docs/BENCHMARKING.md)).
  - Compare only the **same prompt set**, the **same sampling** and the **same token budget**.
  - Report ms/step and tok/step as well as tok/s: tok/s swings by ±5% with acceptance on single prompts.
  - Do not compare code/JSON/"anchor" numbers with ordinary prose.
- **Recovery.**
  - `./stop.sh` stops the container and the watchdog; then run `./run.sh` again.
  - If the host runs low on memory, `scripts/memwatch.sh` stops the container on its own.
  - Nothing outside the container, the repo's `.state/`, the HF cache and the PLE cache is written.

### Running under llama-swap

`deploy/llama-swap-start.sh <port>` is a llama-swap `cmd` adapter.

- It starts vLLM on an internal port, installs E46, and only then opens `<port>`, so no request reaches a half-installed
  server.
- `deploy/llama-swap-stop.sh` is the matching `cmdStop`.
- See `deploy/llama-swap.example.yaml`. The default served name is `qwen3.8-flash-next-optimized`.
- Run `./run.sh` (or `scripts/prepare.sh`) once first to build the local artifacts.
- The model needs ≈100 GiB, so make it exclusive with other large models (for example through a matrix "alone" set).

## Capabilities and validation status

| Capability | Status | Evidence |
|---|---|---|
| Tool calling (single, multi-argument + follow-up turn) | pass | `bench/gates.py` on the full E42 stack (CK42) and on a clean checkout (E42). E43 changes only the sampled draft noise key, not greedy/tool paths |
| Strict JSON and schema-guided JSON | pass | same gates |
| Reasoning (thinking on, `qwen3` parser) | 4/4 pass | same gates |
| Code generation + execution | pass | same gates |
| Perplexity (7,601 human-written tokens) | 2.3787 on the E42 stack (BF16 reference band 2.383–2.400) | CK42 |
| Sampled-distribution correctness | E43 fixes a draft/verify noise coupling that biased E41/E42 sampled output (see [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) §15). Synthetic kernel test: E43 within sampling noise. Served rescore: no significant difference from the greedy-draft reference | CK1 |
| 262,144-token context | configured; needle correct at 108k and 201,650 prompt tokens (E22) | E22 |
| Two concurrent long conversations | 144,047-token needle correct while a 150,053-token conversation decodes at 52.4 tok/s; min MemAvailable 12.4 GB | CK42 (E42 stack) |
| KV capacity | 1,115,942 tokens = 4.26× a 262k request | E42 launch log |
| Two concurrent streams (S=2) | greedy 78.7 / T1.0 72.8 tok/s aggregate on ordinary prose | CK42 |
| Long-context decode speed | 150k context: greedy 46.8–50.9, T1.0 47.0–47.6 tok/s; acceptance flat from 1.5k to 150k | CK42, L1 |

Speed was optimized for S=1 and S=2. `max-num-seqs` is 4; higher concurrency was not a target and is not characterized.
Vision/video inputs (supported by the base recipe) were not evaluated by this work.

## Safety, memory and network exposure

- **Memory.** The container is capped at `--memory 100g --memory-swap 100g`. `serve.sh` refuses to start with less than
  104 GiB `MemAvailable`. Stop every other GPU/LLM server first. On unified memory an exhausted pool can hang the host
  instead of raising OOM, so `serve.sh` also starts Mia's `scripts/memwatch.sh` watchdog, which stops the container when
  `MemAvailable` stays below 6 GiB (or `MemFree` collapses). With the stack installed, observed minimum `MemAvailable`
  during two concurrent long conversations was 12.8–13.5 GB.
- **Do not expose this server to untrusted networks.** The container uses `--network host` and vLLM listens on
  `0.0.0.0` with no API key. `VLLM_SERVER_DEV_MODE=1` is required because the optimization stack is installed through
  vLLM's `/collective_rpc` endpoint, and the worker extension's `exp_exec` method executes a Python file inside the
  worker process. **Anyone who can reach the port can run code in the container.** Bind it behind a firewall, reach it
  over an SSH tunnel, or put an authenticating proxy in front of it.
- The container runs with `--ipc host`, `SYS_NICE` and `SYS_PTRACE`, as in the base recipe.
- The PLE zero-copy path uses a 5 MB shared-memory file `/dev/shm/exp_ple_zc` inside the container.

## Documentation

| Document | Contents |
|---|---|
| [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) | every retained optimization, in install order: mechanism, measured effect, correctness class |
| [docs/RESULTS.md](docs/RESULTS.md) | result tables with exact settings and measurement type; quality and long-context validation |
| [docs/BENCHMARKING.md](docs/BENCHMARKING.md) | how to measure: metrics, suites, same-process A/B, pitfalls |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | the model, the Mia baseline path and the modified path; step anatomy |
| [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) | curated history including rejected experiments and lessons |
| [docs/DECISIONS.md](docs/DECISIONS.md) | decisions (e.g. D-S6) |
| [docs/PORTING_PLAYBOOK.md](docs/PORTING_PLAYBOOK.md) | lessons classified as GB10-general, spec-decoding-general, Qwen3.8-specific, checkpoint-specific |
| [CHANGELOG.md](CHANGELOG.md), [THIRD_PARTY.md](THIRD_PARTY.md) | release notes; licensing map |

## Credits

- **[MiaAI Lab](https://x.com/MiaAI_lab)** — the [single-DGX-Spark recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
  this work starts from (launch profile, patched vLLM model files, PLE CPU offload with the memory-mapped packed table,
  the packed-table builder, the `memwatch.sh` watchdog, the chat template as shipped) and the
  [`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4) checkpoint.
  MiaAI Lab's later `improve/single-spark` branch independently ships a fix equivalent to our PLE row fix.
- **Qwen / Alibaba** — the [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) base model and its MTP head.
- **The vLLM project** — the serving engine; `files/vllm/**` and four overlays are modified vLLM source (Apache-2.0).
- **FlashInfer and NVIDIA CUTLASS** — the NVFP4 fused-MoE kernels that carry the routed experts.
- **[lancelind/qwen3.8-Flash-DGX](https://github.com/lancelind/qwen3.8-Flash-DGX)** — the FP8-KV-cache approach for the QSA
  attention backend, inherited through Mia's recipe (Mia's patched QSA files used here carry it; Mia's README credits it).
- Ideas from public work cited in the research log: vLLM PR #54651 (small-batch top-k/top-p), FastMTP-style drafter
  self-distillation, and the FP8 draft-head tactic from Mia's recipe PR #31.

## License

This repository is licensed under the **GNU Affero General Public License v3.0 or later** (see `LICENSE`, the same text
as Mia's recipe), except for files that carry an Apache-2.0 SPDX header (vLLM-derived files; see `LICENSES/` and
[THIRD_PARTY.md](THIRD_PARTY.md)). If you modify this code and offer it as a network service, AGPL section 13 requires
offering your users the corresponding source. **Model weights are not included**; they are governed by the checkpoint's
own license and the upstream Qwen terms.
