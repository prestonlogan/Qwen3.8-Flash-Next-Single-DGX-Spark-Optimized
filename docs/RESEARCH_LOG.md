# Research log (curated)

This log is a curated history of the work that produced the E42 stack. It includes the dead ends, because those are often the useful part. It is condensed from the full working notes: about 1,400 lines of per-experiment records and a status file, kept outside this repository.

Unless stated otherwise, all numbers are:

- single stream (S=1);
- ordinary prose;
- 800 tokens;
- thinking off;
- on one DGX Spark.

"Direct A/B" means a same-process on/off comparison (see [BENCHMARKING.md](BENCHMARKING.md)).

> **Naming note.** Experiment numbers were assigned sequentially. From round 25 onward, "E39"–"E42" were reused for *stack milestones*, and the earlier experiments with those numbers were rejected probes:
> - E39: shared-expert gate fusion;
> - E40: HC tail re-test;
> - E41: MoE scale buffer;
> - E42: PLE W8.
>
> This log always says which one it means.

## Phase 0 — Baseline and the PLE bug (E01, E10–E11)

- **E01.** Isolated, production-identical launch of the Mia recipe. The anchor (hash-map) prompt gave 49.62 tok/s, which matches the 48.7 sparkDash reference. The five ordinary prose prompts gave **36.66 tok/s** (2.18 tok/step, ≈59.5 ms/step). Every later "vs E01" claim uses this ordinary-prose baseline.
- **E02.** Pinning vLLM threads to Cortex-X925 cores: no change (±1%). Host CPU speed is not on the S=1 critical path.
- **E10 → E11.** While building a GPU-side PLE gather, we found that the CPU-offload PLE path delivered **only token row 0** of each forward to the model: a strided-copy bug in the packed-mmap branch.
  - Teacher-forced PPL on 7,601 human-written tokens: **4.998 → 2.382** once fixed.
  - The fix (`ple_layer_fixrows.py`) is speed-neutral.
  - It is part of this stack. MiaAI Lab's later branch ships an equivalent fix.

## Phase 1 — Heads and draft vocabulary (E03–E09, E12, E15)

- **Draft head (E03/E04).** An FP8 per-row draft head (Triton W8A16 GEMV), then a dedicated 64,877-row reduced draft vocabulary. It is draft-only, so outputs stay exact.
- **Target head (E06–E09, E12).** The approach: a cheap coarse head selects candidates, then an exact BF16 refine recomputes their logits.
  - A certified FP8 head using Cauchy–Schwarz bounds failed; the bound is ~100× too loose.
  - A plain FP8 head flips 2.4% of argmaxes, which is not acceptable.
  - Containment works: the exact argmax sits at rank ≤5 in the approximate ordering.
  - INT4-g32 coarse + BF16 refine (C=256) was adopted.
- **E05.** Upstream PR #55054 (async H2D in the PLE short-conv metadata) had no measurable effect on GB10 at S=1.

## Phase 2 — Dense weights (E13, E18, E22, E45)

- **E13.** HC mixers W8 g64 through our own Triton weight-only GEMV: ms/step 55.1 → 52.0.
  - Quality-screened by PPL, top-1 agreement and gates.
  - Prior art agrees: FP8 *GEMM* HC is null on vLLM, and the win needs a weight-only GEMV.
- **E18.** Draft dense Linears W8: draft pass 540 → 186 µs.
- **E22.** Target dense projections (GDN in_proj_qkvz/out_proj, QSA qkv/o) moved from MXFP8 to **W6A16 g32**: ordinary prose +4.3%, and 2.76 GB freed.
  - int6 passed the gates (PPL 2.393/2.398, agreement at the noise floor).
  - int5 (PPL 2.414) and int4 (2.541) were rejected.
- **E45 (rejected).** Draft dense W8 → W6: too small for a numeric change.

## Phase 3 — PLE delivery (E20, E35, E36)

- **E20.** An async PLE handshake (in-graph host-flag spin) cut GPU idle time from 2.45 to 0.55 ms.
- **E35.** Zero-copy PLE rows: the CPU worker writes the rows straight into host-mapped shared memory, and the graph copies them after the flag. −1.0 ms/step, rows bit-exact.
- **E36.** The PLE request moved to a shared-memory slot that the worker spin-polls (zmq kept as fallback): −0.57 ms/step.

## Phase 4 — Draft graphs, MoE tactic, head refinements (E26–E34)

- **E26a/b.** Exact-size draft decode graphs 1..4, plus a draft-prefill MLP on the last rows only: propose 5.32 → 4.38 ms. Draft-only and exact.
- **E30.** FlashInfer CUTLASS fused-MoE gemm1 tactic override at M=4: bitwise identical, −0.4 to −0.7 ms.
  - Non-identical tactics were rejected on principle (E30b, E43): a different reduction order is not the reference path.
- **E32 → E34.** Draft head: INT4 coarse + FP32 refine top-8 (−0.83 ms), then INT2 coarse + refine top-64 (−0.70 ms). Both exact up to ties.
- **E33.** Target head coarse INT4 → **INT2 g16**, argmax verified on 5,066 rows: −0.94 ms.
- **Rejected in this phase:**
  - **E25:** custom fused NVFP4 MoE kernel. CUTLASS is already at the bandwidth ceiling for this access pattern.
  - **E27:** PDL on our own GEMVs. The apparent gain came from skipping dependencies; the race-free variant gives nothing. *Lesson: never trust a speed result without the tok/step and agreement check.*
  - **E28**, and its later re-test: fused HC tail, null.
  - **E29:** W6 bandwidth audit. The configs were already near-optimal.

## Phase 5 — Small kernels (E37, E38) and the hardware-limit classification (P43–P54)

- **E37.** Triton BF16 GEMV for the router gate and GDN in_proj_ba: −0.36 ms. Same precision; the router output is closer to FP32 than cuBLAS.
- **E38.** Fused HC split-K reduce+cast: −0.44 ms, bitwise identical.
- **Rejected:**
  - shared-expert gate fusion (old "E39"), null;
  - NVFP4 activation-scale buffer (old "E41"), null;
  - PLE key/value W8 (old "E42"), too small for a numeric change;
  - fused draft refine (E44), ≈0.05 ms;
  - L2 prefetch side stream (E48), no gain.
- **P44–P53 classification.**
  - Routed MoE touches ≈25 distinct experts per 4-token verify window, streaming at ≈205 GB/s against a measured practical peak of ≈243 GB/s.
  - The byte-streaming kernels (MoE, W6 dense, W8 HC, heads) make up ≈40 of the ≈44.6 ms step, at 84–95% of practical peak.
  - The critical path (P47) is entirely bandwidth-bound streams.
  - A persistent-executor/megakernel was falsified (P52): idle gaps are ≈0.08 ms/step, so "7 ms of executor overhead" does not exist.
  - **Verdict:** hardware-limited for this execution strategy. The remaining exact kernel upside was estimated at ≈2–3 ms/step.
- **P54.** A custom CUDA NVFP4 decode MoE ("moedec", sm_121a) reached **parity** with CUTLASS, not a win. Closed.
- **P51.** Early rejection of speculative positions inside the verify pass via a logit lens: falsified.
- **E17.** Confidence-gated position cancellation: ≤+1.3% simulated. vLLM adaptive verification does not support GDN.

## Phase 6 — Correctness audits (Q3, F1, B8, V39)

- **Q3.** In sampled mode, the approximate fast target head (INT2 + refine C=256) can change the *processed* sampling distribution.
  - Verification does not correct this, because it changes the target, not the draft.
  - **Decision: the E39 exactness gate.** The fast head is used only for all-greedy batches without logprobs, penalties or grammar. Everything else uses the exact BF16 head.
- **F1.** Audit of MTP implementation fidelity against the checkpoint semantics (router, norms, HC input, KV): no fidelity loss found.
- **B8.** A latent autotuner bug in the E37 in_proj_ba install: an invalid split-K config could win the timing race on relaunch.
  - Fixed with a validity check plus a relerr gate.
  - No measured result was affected.
- **V39.** Periodic full verification (PPL, gates, dual long context) on the E39 stack.

## Phase 7 — Drafter self-distillation (T8) — E40

- **Ceiling first.** Acceptance anatomy (P50) showed 71% / 64% / 50% conditional acceptance by draft position. A perfectly recursion-trained head caps at ≈+7.5% at MTP3; the model card says the MTP layer was already trained multi-step.
- **T8 cheap test.** A LoRA r32 on the 4B MTP drafter, trained on self-generated prose (≈80k tokens, ~4 min on the Spark), with the target model untouched.
  - The first two-launch A/B said +6.0%. The **same-process** check said **+3.6%**: launch-to-launch variance had inflated the first number.
  - Sealed PROSE2: +3.4%. T0.7: +5.1%. T1.0: +2.7%. It also transferred to code/JSON, 150k context and S=2.
  - Promoted as **E40** (the r32a adapter).
- **T8 round 2 (closed).** ×6 data (r32b LoRA) and full fine-tuning each gained only ≈+1% offline.
  - Served: r32b −2.3%; full1 +0.0% ±0.6 on PROSE2.
  - *Lesson: teacher-forced offline gains of a few percent do not transfer. More teacher-forced data and capacity is a closed route. A future drafter lever needs a served/on-policy signal and a cheap rejection test first.*
- **S5 (closed).** Fine-tuning for sampled acceptance with a TV-style loss: +0.3% offline.

## Phase 8 — The sampled path (S1–S7) — E41, E42

Chat clients sample (the generation config is T1.0 / top-p 0.95 / top-k 20), and sampled decode was the slow path.

- **S1.** Block verification (lossless) cut tok/step by 2.5%. Rejected.
- **S2/S3 → E41.** Probabilistic drafting.
  - The draft distribution q comes from the fast draft head: top-20, top-p 0.9, temperature ×0.9.
  - vLLM's rejection sampler keeps the output exactly target-distributed for any q.
  - Direct A/Bs: PROSE T0.7 **+5.0%**, S=2 +7.2%, 150k T1.0 +4.9%. The PROSE2 T1.0 figure of ≈+14.7% is a **chained** estimate from two A/Bs.
  - Greedy is unaffected. Code/JSON is neutral.
- **S6 (not promoted).** An approximate fast target head for sampled requests (INT2 + exact refine of the top 1,024–2,048).
  - Result: zero changed rows among 8,192 captured sampled rows, and a direct A/B of **+8.2%** on the dev set.
  - Exactness cannot be certified, and verification does not correct target-head changes.
  - Left disabled as user decision **D-S6** ([DECISIONS.md](DECISIONS.md)).
- **S7 (closed).** Certified-exact head via INT4 + low-rank correction + Cauchy–Schwarz bound: still a median of 2.35k and a p99 of 142k candidates, so impractical. This rules out only that family of bounds.
- **L1.** Long-context acceptance: greedy tok/step is flat from 1.5k to 150k tokens of context (2.31–2.36), so there is no long-context gap to chase.
- **U30.** vLLM v0.30.0 source-diff audit. None of the release's Qwen3.8 performance PRs are in the pinned image.
  - Most items are superseded by this stack, prefill-only, or ≤0.5 ms/step combined.
  - External evidence shows v0.29 → v0.30 warm decode unchanged.
  - **No rebase.** The one new mechanism was the small-batch top-k/top-p fix (PR #54651-style).
- **U30a → E42.** Below batch 8, the pinned sampler fully sorts the 248k vocabulary (1.02 ms per call, 2 calls per sampled step).
  - The replacement: `topk(64)`, reordered to the stock tie order, then threshold, top-p and scatter.
  - 0/2,048 real rows differ, tie order included.
  - Direct A/B: PROSE2 T1.0 −0.96 ms/step (12/12 prompts faster); fresh PROSE3 T0.7 −0.91 ms/step (8/8). Promoted as **E42**.
- **H1/H2 (parked).** Low-rank coarse target head (<1% ceiling); lossless 12-bit BF16 head compression for the sampled path (≈1–2% realistic ceiling).

## Incidents and process lessons

- **Container exits** during probes (rounds 20–21) led to three rules:
  - probes free their stashed tensors before recapturing;
  - "off" paths undo every wrapper they ever installed;
  - relaunch after many graph recaptures, which leak graph-pool memory.
- **Measurement hygiene:**
  - same-process toggles for every A/B;
  - paired per-prompt analysis;
  - ms/step for step-cost changes;
  - labels that separate direct from chained numbers;
  - a dev set (PROSE2) kept separate from a fresh confirmation set (PROSE3).
- **Gating rule.** Offline or micro-benchmark gains under ≈5% rarely survive serving, so kernel work needs a ≥5% offline estimate first.
- **Tracked jobs** (`bench/xjob.sh`) replaced `pgrep -f` waits, which could match themselves.
