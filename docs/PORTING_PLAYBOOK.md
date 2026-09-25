# Porting playbook

These are lessons from ~60 optimization rounds on Qwen3.8-Flash-Next NVFP4 on one DGX Spark. Each lesson is tagged by
how far it is likely to transfer. The numbers are from this project's measurements (see RESEARCH_LOG.md / RESULTS.md).

## GB10-general (any model on DGX Spark / SM121 unified memory)

- **Single-stream decode is DRAM-bound.** The practical peak is ≈243 GB/s. Before optimizing a kernel, compute bytes ÷ 243
  GB/s. Well-written decode kernels reach within ≈10% of it; below that there is little left to gain.
- **Time kernels DRAM-cold.** L2-hot microbenchmarks overstate bandwidth by as much as 2× (we once "saw" 527 GB/s). Rotate
  ≥12 weight copies.
- **Small-M dense GEMV:** at M≤8, cuBLAS is not bandwidth-optimal. Triton W8A16/W6A16 group-quantized GEMVs pay off
  because they move fewer bytes. W5 did not beat W6 at the cold-DRAM level, because unpacking cost ate the byte savings.
- **Use full CUDA graphs for decode** (`FULL_DECODE_ONLY`). After that, host overhead is ≈0.3–0.7 ms/step, and removing
  host syncs is worth ≤1%. Persistent megakernels save ≈0.15 µs per dependency, which is not worth it here.
- **Unified memory can hang the host instead of raising OOM.** Cap the container, run a MemAvailable watchdog, and keep
  in-server transient allocations small (≤1.5 GB near the watchdog floor). Run config sweeps in side containers.
- torch.compile/inductor bought nothing on top of hand-written graph-captured kernels in this build.

## Speculative-decoding-general

- **Optimize tok/step × ms/step, not one alone.** MTP2 saved ≈3 ms/step but lost ≈10% tok/step, so it was a net −2 to
  −4.5%.
- **Ordinary prose is the hard case.** Code, JSON and "technical explainer" prompts accept far more drafts. Never compare
  across workloads.
- **Drafter distillation on self-generated text** (a LoRA on the MTP head) gained +3.6–5% at zero verify cost.
- **Fast approximate heads on the *draft* side are free for quality.** The target verifies every token. We used INT2
  coarse plus FP32 refine top-64 on a 64.9k-id draft vocabulary.
- **Sampled drafting must use independent noise.** If the draft Gumbel key equals the rejection sampler's residual key,
  the output distribution is biased. We fixed exactly that bug (E43). Test synthetically for TV.
- **The target head for greedy** can use INT2 coarse plus exact BF16 refine over a shortlist (exact argmax with fallback).
  Sampled requests need the full distribution: compress the weights losslessly (HX2) instead of approximating them.
- Early draft stopping (dynamic depth) is only a win with a GPU-side variable-width verify. Otherwise the fixed-shape
  graph costs the same.

## Qwen3.8 architecture-specific (Qwen3.8-Flash-Next family)

- The PLE (per-layer embedding) table is ≈27 GB. CPU offload through a packed mmap table (Mia's design) is what makes the
  model fit. Zero-copy rows plus a shared-memory request slot and an in-graph flag spin hide it almost completely.
- The HyperConnection (HC) mixers and the GDN `in_proj_ba` / router gate are small GEMVs that cuBLAS handles poorly. Triton
  GEMVs and a fused split-K reduce helped.
- QSA indexer top-k reuse across MTP steps (IndexShare) is already in the day-0 fork.
- The routed-expert MoE at M=1..8 is the largest share (≈14 ms of ≈36 ms verify). A dp4a W4A4 decode MoE (qwenfast
  `qf_moe`) beat FlashInfer CUTLASS by ≈+4% end-to-end.
- No expert pruning, no top-k reduction: all of the gains above come without quality relaxation.

## Checkpoint-specific (Mia-AiLab/Qwen3.8-Flash-Next-NVFP4 @ 925d7be)

- The r32a drafter LoRA, the draft vocabulary `dv_cur_u48k.txt`, and the W6/W8 quantized copies of the dense projections
  are all derived from this checkpoint. Rebuild and re-gate them for any other checkpoint.
- The PLE row-fix (B1) addresses a bug in the recipe version we started from. MiaAI Lab's later branch ships an
  equivalent fix.
- The HX2 exponent coding relies on this head's BF16 exponent statistics (0.01% escapes). Re-measure the escape rate
  before reusing it.
