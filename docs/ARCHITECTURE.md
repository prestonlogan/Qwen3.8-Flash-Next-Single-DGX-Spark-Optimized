# Architecture: the Mia baseline path and the optimized path

This document explains what happens in one decode step, what MiaAI Lab's recipe does, and what this repository
changes. Numbers come from the experiment records (ids in brackets); they are per-step GPU/host timings for a single
stream unless noted.

## 1. The model, as served here

`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` is a hybrid mixture-of-experts language model (plus vision, not used here).
The parts that matter for decode speed:

| Component | What it is | Precision in the checkpoint | How it is served (Mia) |
|---|---|---|---|
| GDN layers (36) | gated linear-attention ("Gated DeltaNet") layers with a recurrent state instead of a KV cache | MXFP8 projections | CUTLASS MXFP8 GEMMs; BF16 recurrent state |
| QSA layers (12) | sparse-attention layers with a learned indexer that selects which KV blocks to attend to (2,048-token indexer budget) | MXFP8 projections | Triton QSA kernels, FP8 KV cache |
| Routed MoE (48 layers) | 512 experts, top-10 routing, plus a gated shared expert | NVFP4 (routed), MXFP8 (shared) | FlashInfer CUTLASS fused MoE (W4A4) |
| Hyper-connections (HC) | 4-branch residual streams with learned mixers around every block (97 mixer modules in the target) | BF16 | cuBLAS GEMMs + small Triton kernels |
| PLE | per-token lookups into a very large hashed n-gram embedding table, combined into the residual stream | NVFP4 codes + FP8 scales | offloaded to a CPU worker process; memory-mapped packed table (~27 GiB) in the page cache |
| LM head | 248,320 × 2,560 | BF16 (1.27 GB) | eager full-vocab GEMV every step |
| MTP head | one Multi-Token-Prediction layer (HC, attention, MoE) used as the drafter | BF16 dense, NVFP4 experts | MTP speculative decoding with 3 draft tokens; Marlin W4A16 draft MoE; reduced 47,149-token draft vocabulary |

The routed experts dominate the bytes: each expert is ≈2.76 MB, and the 4 tokens of one verify step touch ≈25
distinct experts per layer on real prose (P44). Every byte streamed from LPDDR5X costs time; the measured practical
peak on GB10 is **243 GB/s** (P44).

## 2. One decode step

Speculative decoding with MTP (3 draft tokens):

1. **Verify.** The target model runs one forward over 4 rows (M=4): the last accepted token plus 3 draft tokens. This
   is a captured CUDA graph ("FULL_DECODE_ONLY").
2. **Target head + sampling.** Logits for the 4 rows; the rejection sampler accepts a prefix of the draft tokens and
   emits one more token from the target. Greedy requests accept a draft token only if it equals the target's argmax;
   sampled requests use standard speculative (rejection) sampling, which leaves the output distribution equal to the
   target's for any draft distribution.
3. **Propose.** The MTP drafter runs a draft-prefill over the accepted positions and then 3 recursive draft steps, each
   scoring its reduced draft vocabulary.

Speed is **(committed tokens per step) / (time per step)**. Step time is almost content-independent (E01: 59–61 ms
across prose, code and JSON), while tokens per step depends strongly on content: ≈2.2–2.5 on ordinary prose, ≈3.0 on
the hash-map "anchor" prompt, ≈3.4 on code, ≈3.9 on counting (E01, T8).

## 3. The Mia baseline path (E00/E01)

Production-identical isolated launch of the recipe as we started (E01): 59.5 ms/step, 2.18 tok/step, **36.66 tok/s**
on ordinary prose. An earlier profile of the same configuration (E00, 62.35 ms median step) showed:

| Phase | Time | Notes |
|---|---|---|
| Verify graph | 43.5 ms | NVFP4 MoE ≈16.9, MXFP8 dense GEMMs ≈8.6 + GDN 3.7, BF16 HC ≈7.5 ms, routers, small kernels |
| Target LM head | 5.3 ms | full BF16 248k-row GEMV, eager |
| Propose | ≈8.5 ms | 3 draft graphs of ≈2.8 ms; the BF16 47k draft head alone 1.3–1.5 ms per draft step |
| GPU idle | ≈5 ms | mostly the host waiting on the PLE CPU worker (handshake), plus launch gaps |

In that recipe version the PLE offload also had a correctness bug: only row 0 of each forward received its PLE
embedding (perplexity 4.998 vs 2.382 fixed). See section B1 of [OPTIMIZATIONS.md](OPTIMIZATIONS.md).

## 4. The optimized path (E42)

```
                 ┌──────────────────────────────── verify graph (M=4, CUDA graph) ───────────────────────────────┐
 accepted token ─► layer 0 … PLE spin (≈6 µs) … 48 × [ GDN/QSA with W6A16 projections │ W8 HC mixers + fused     │
 + 3 drafts        (rows already in host-shared    │   split-K reduce │ router: Triton BF16 GEMV │ NVFP4 routed MoE │
                   memory from the PLE worker)     │   (CUTLASS, tactic 16/49) + MXFP8 shared expert (aux stream) ]    │
                 └───────────────────────────────────────────────────────────────────────────────────────────────┘
                                   │
            target head ───────────┤ greedy-only batch → INT2 coarse (238 MB) → exact BF16 refine of top-512
                                   │ anything else     → original BF16 full head (exact)
            sampler ───────────────┤ top-k/top-p: topk(64) fast path when 1 ≤ top_k ≤ 64 (bit-identical)
                                   ▼
            propose (MTP, r32a weights, W8 dense, exact-size graphs 1..4, draft-prefill MLP on last rows only)
               draft head: INT2 coarse over 64,877 rows → FP32 refine of top-64 from FP8 rows
               greedy request → argmax draft        sampled request → probabilistic draft q (top-20, top-p 0.9, T×0.9)
```

### Dense projections: MXFP8 → W6A16 (quality-gated)

The 96 GDN/QSA projection matrices move from MXFP8 (1.03 B/param) to 6-bit RTN weights with BF16 activations
(0.8125 B/param) and a Triton GEMV. This is the one intentional numeric change to the target's main path; it passed
the perplexity band, top-1 agreement and functional gates, while 5-bit and 4-bit did not (E22). The HC mixers similarly
move from BF16 to W8 group-64 (E13).

### PLE delivery

Mia's recipe replaced the upstream CUDA stream-memory-op semaphore (unsupported on GB10) with a host handshake: the GPU
worker posts a request, the CPU worker gathers rows from the memory-mapped table and copies them to the GPU, and the
GPU worker waits. Here (E20, E35, E36):

- the host no longer waits; a one-CTA kernel inside the verify graph spins on a host-mapped done flag;
- the CPU worker writes rows directly into host-shared memory (no CUDA copy on the CPU side), fences with `dmb sy`, and
  sets the flag; the in-graph kernel copies the rows to the GPU buffer;
- requests travel through a shared-memory slot that the worker polls, instead of a zmq hop.

The PLE wait inside the step dropped to ≈6 µs, i.e. it is completely overlapped with layer-0 work (E37, P45).

### Target head: coarse → exact refine, and the greedy gate

The exact head reads 1.27 GB per step (≈5.2 ms). For decode-sized batches a coarse INT2 head (238 MB) ranks all 248,320
tokens, the top C candidates are recomputed exactly in BF16, and every other logit is floored below the refined ones.
The exact top-1 was always within coarse rank 17 on real rows (E33), so the argmax is exact up to exact BF16 ties. The
coarse head is not distribution-exact (top-20 containment can fail on flat rows), so the **E39 gate** uses it only when
every request in the batch is greedy with no logprobs, penalties, bias, bad words, thinking budget or grammar; all other
batches use the original head. The gate uses C=512.

### Drafting: reduced vocabulary, cheaper head, better weights, probabilistic q

- The draft vocabulary is 64,877 ids (Mia's 47,149 plus all ids < 48k): ≈12% of ordinary-prose tokens were outside the
  47k list and could never be drafted (E03).
- The draft head is INT2 coarse over those rows, then an FP32 refine of the top-64 from per-row FP8 weights: equal to
  the FP8 head up to ties (E34).
- The drafter weights are the r32a self-distilled patch (T8): +4.7% tokens/step on greedy prose, target untouched.
- For **sampled** requests, the drafter samples from a shaped distribution *q* (top-20 of the fast draft logits, draft
  top-p 0.9, temperature × 0.9) instead of drafting its argmax. vLLM's rejection sampler uses exactly that *q*, so the
  output distribution is still the target's; acceptance rises because Σ min(p, q) > p(argmax) when the target is sampled
  (S1–S4).

### Sampler: top-k/top-p fast path

With fewer than 8 rows the pinned engine applies top-k/top-p with a full sort of the 248k vocabulary (1.02 ms, twice
per sampled step). For 1 ≤ top_k ≤ 64 a `topk(64)`-based path with the stock tie order gives identical results in
0.27 ms (E42).

### Hot-install mechanism

The model is loaded once with the boot-time overlays (PLE row fix, FP8 draft head + draft vocabulary, r32a weights, PLE
zero-copy worker, worker extension). Everything else is installed into the live process:

1. `serve.sh` starts vLLM with `--worker-extension-cls exp_q38_ext.ExpWorkerExt` and `VLLM_SERVER_DEV_MODE=1`, and
   mounts `overlays/runtime/` read-only at `/exp` in the container.
2. `install_stack.sh` writes a control file (`overlays/runtime/<name>.txt`, e.g. `head2.txt` = `on 256`) and calls
   `POST /collective_rpc` with method `exp_exec` and the path of an `exec_*.py` script (`bench/rpc.py`).
3. `exp_exec` runs that script inside each worker with access to the live model runner. The script replaces module
   methods or weights, and, where the verify/draft graphs are affected, clears and recaptures the CUDA graphs.

Every script supports `off` (or an equivalent restore mode) so it can be toggled in the same process, which is what makes
the same-process A/B methodology in [BENCHMARKING.md](BENCHMARKING.md) possible. The same endpoint is why the server must
not be exposed to untrusted networks.

## 5. Step anatomy on the optimized stack

E41 stack, anchor prompt, 600 tokens (S6 record); E42 only changes the sampled "head + sample" phase.

| Phase (ms) | Greedy | Sampled T1.0 / p0.95 / k20 |
|---|---|---|
| Execute (whole step, host view) | 42.2 | 47.5 |
| Verify graph (GPU) | 38.1 | 38.5 |
| Propose (GPU) | 3.43 | 3.45 |
| Target head + sample | 1.36 | 6.43 |
| Idle | 0.52 | 0.55 |

The sampled path is ≈5 ms slower per step mainly because the gate sends it to the exact 1.27 GB head (5.19 ms at ≈245 GB/s,
U30). E42 then removed ≈0.9 ms of full-vocabulary sorting from sampled steps (measured 49.9 → 49.0 ms/step on PROSE2 at T1.0).

What is inside the verify graph (real routing, P53, E38-era stack; 41.2 ms span, 0.08 ms idle):

| Class | ms | Bandwidth |
|---|---|---|
| Routed MoE GEMMs (CUTLASS NVFP4) | 17.96 | ≈205 GB/s at 25 distinct experts/layer |
| W6 dense projections | 11.37 | 183–223 GB/s |
| W8 HC mixers | 4.53 | 165–225 GB/s |
| MoE routing prep/finalize | 1.38 | — |
| MXFP8 shared expert (aux stream) | 1.33 | — |
| Router / `in_proj_ba` GEMVs | 1.06 | ≈220 GB/s |
| GDN, QSA, HC small kernels, ATen glue, head | ≈3.5 | — |

## 6. Why the remaining step is "hardware-limited for this strategy"

- The byte-streaming kernels (MoE, dense, HC, heads) are ≈40 ms of a ≈44.6 ms step and run at 84–95% of the 243 GB/s
  practical peak (P44, P45). The critical path of the verify graph consists of these streams; small ATen/prep kernels
  contribute ≈0 to it (P47).
- A custom NVFP4 MoE kernel (E25: 181 GB/s; P54 "moedec": parity standalone, +0.7 ms/step slower in-graph), persistent
  executors (P52: ≤0.3 ms/step), more fusions (E28, E39, E40: null) and PDL (E27) did not beat the current kernels.
- The largest remaining exact headroom is therefore not kernel speed but **committed tokens per weight read**: at
  2.3 tok/step the stack reads ≈3.3 GB per committed token (ceiling analysis). Acceptance is bounded by the MTP head's
  quality (33% of first rejections are tokens the draft ranked below 4th, P50), which is why the drafter weights (T8) and
  the sampled draft distribution (E41) were the late levers.
- The remaining quality-preserving kernel ceiling was estimated at +7–10% in aggregate, spread over several ≤2 ms items
  that each need a kernel beating CUTLASS/Triton rooflines.
