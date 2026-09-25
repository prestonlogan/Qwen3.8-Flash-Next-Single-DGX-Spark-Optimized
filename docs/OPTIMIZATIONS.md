# Optimizations in the promoted stack (E42)

This document lists every optimization that is active in the promoted configuration **E42**, in the order it is
applied: first the boot-time items that `scripts/serve.sh` mounts into the container, then the hot-install steps of
`scripts/install_stack.sh` in their exact order. Experiment ids (E.., S.., T8, U30a) refer to
[RESEARCH_LOG.md](RESEARCH_LOG.md) and the original experiment records.

## How to read the "effect" column

| Label | Meaning |
|---|---|
| **alternating A/B** | on/off arms interleaved in one loaded server process (e.g. on/off/on/off); the most reliable form |
| **sequential** | measured after the previous item on the same load or a relaunch; includes launch/session drift |
| **microbench** | isolated kernel timing (CUDA-graph replay), not end-to-end |
| **chained** | product of two separately measured A/Bs; an estimate |

Unless noted, end-to-end numbers are single-stream (S=1), 800 tokens, greedy, on the PROSE "ordinary-5" prompts. The
absolute tok/s values of early items (E04–E26) belong to their own point in the history and are not comparable with
the final numbers; the per-item **ms/step** deltas are the robust quantity.

## Correctness classes

| Class | Meaning |
|---|---|
| **correctness fix** | removes a bug; makes the served model match the reference |
| **bitwise identical** | same outputs bit for bit (checked on real layers) |
| **output-exact (draft-only)** | changes only the drafter; the target verifies every token (greedy exact-match, or rejection sampling), so outputs cannot change except via the engine's normal numeric nondeterminism. Acceptance (speed) may change. |
| **exact up to ties** | the result equals the reference except where two candidates have exactly equal values and a different one is picked |
| **distribution-exact** | standard speculative (rejection) sampling with the draft distribution *q* that was actually sampled: the output distribution equals the target's for any *q* |
| **quality-gated numeric change** | the target's arithmetic changes; accepted only because perplexity, top-1 agreement and functional gates stay inside the reference noise band |

Background noise: greedy decoding on this engine is not bitwise reproducible between identical runs (batch/graph shape
effects and FlashInfer's fused-finalize atomics), so exactness is checked with in-server counters and row
comparisons, not with text diffs. The perplexity reference band for the BF16/MXFP8 model is 2.383–2.400
(7,601 human-written tokens); identical-run top-1 agreement is only 93.66%.

---

## A. Boot-time items (`scripts/serve.sh`)

These are bind-mounted into the container or set by environment variable, so they are active from model load.

### B1. PLE row-delivery fix (E10/E11) — `overlays/runtime/ple_layer_fixrows.py`

- **Problem.** In the recipe version we started from, the packed-table branch of the CPU-offloaded PLE layer
  (`files/vllm/ple_layer_patched.py`, `forward_impl`) wrote PLE rows through a *strided* view of the output buffer.
  `reshape` of that view is a copy when a forward has more than one token, so `index_select(..., out=copy)` wrote into a
  temporary and **only token row 0 of each forward received its PLE embedding**; all verify positions 1–3 and all but the
  first prefill token of each chunk got zeros.
- **Fix.** Gather contiguously, then `copy_` into the strided view.
- **Effect.** Teacher-forced perplexity 4.998 → **2.382** (GPU-side reference path 2.387). 1,015 validated token rows,
  0 mismatches. Speed-neutral: ordinary prose 39.45 → 39.44 tok/s, 58.0 → 57.8 ms/step (sequential, E04 vs E11).
- **Class.** correctness fix. MiaAI Lab's later `improve/single-spark` branch ships an equivalent fix.

### B2. FP8 draft head over a 64,877-token draft vocabulary (E03/E04) — `mtp_exp_t8.py`, `dv_cur_u48k.txt`

- **Mechanism.** The MTP drafter only scores a reduced vocabulary. Mia's recipe used a 47,149-id list; we use
  `dv_cur_u48k` = that list ∪ all ids < 48k (64,877 ids), because ~12% of ordinary-prose tokens fell outside the 47k list
  and are guaranteed draft misses. The draft head rows are stored as per-row FP8 E4M3 (0.16 GiB) with a Triton GEMV
  (`EXP_MTP_DRAFT_HEAD_FP8=1`); the BF16 slice is dropped.
- **Effect.** Draft head per call 1.45 ms (BF16 47k) → ≈0.70 ms (FP8 65k) (microbench). Ordinary prose tok/step
  2.183 → 2.290 (+4.9%), ms/step 59.61 → 58.03, tok/s 36.66 → 39.45 (+7.6%) (sequential, E04 vs E01; both before the PLE fix).
  Vocabulary sizes 65k–101k were screened; gains saturate at ~65–86k.
- **Class.** output-exact (draft-only). The FP8 head is later replaced by INT2 coarse + FP32 refine of these FP8 rows (E34).

### B3. r32a drafter weights (T8) — `adapters/r32a/`, `EXP_MTP_PATCH`

- **Mechanism.** A rank-32 LoRA on the MTP drafter's dense matrices and norms, trained by FastMTP-style
  self-distillation on 88 self-generated greedy prose sequences (~4 min of training). `scripts/build_adapter.sh` merges it
  deterministically into 22 BF16 MTP tensors (175 MB, SHA-256 `bd3c1807…`); `mtp_exp_t8.py` substitutes them at load.
  The draft-side W8/INT2 installs below quantize from the patched weights. The target model is untouched.
- **Effect (alternating A/B, same process, hot-swapped drafter).** PROSE greedy 53.65 → 55.59 tok/s (+3.6%),
  tok/step 2.405 → 2.519 (+4.7%), 6/6 prompts positive. PROSE2 (then sealed) greedy +3.4% (10/12 prompts);
  PROSE T0.7 +5.1%; PROSE2 T1.0/p0.95/k20 +2.7%; code greedy +2.6%, JSON greedy +6.9%; 150k-context greedy +3.5%;
  S=2 greedy aggregate +4.1%. ms/step unchanged (same kernels and sizes).
- **Class.** output-exact (draft-only); distribution-exact for sampled requests.
- Opt out with `Q38_NO_ADAPTER=1`.

### B4. PLE zero-copy worker (E35/E36 host side) — `overlays/runtime/ple_worker_zc.py`

- Replaces Mia's PLE CPU-offload worker. When the shared buffer `/dev/shm/exp_ple_zc` is enabled by the GPU side
  (step 11 below), the worker writes rows directly into host-shared memory, skips the CUDA copy and stream sync, issues an
  ARM `dmb sy` barrier (`exp_dmb.so`, built by `prepare.sh`), publishes a done flag, and polls a shared-memory request
  slot instead of waiting on zmq. Otherwise, or on any attach error, it behaves like the original DMA path.
- Effect and correctness: see step 11.

### B5. Worker extension — `overlays/runtime/exp_q38_ext.py`

- `--worker-extension-cls exp_q38_ext.ExpWorkerExt` adds `exp_exec(path)`, which runs one `overlays/runtime/exec_*.py`
  script inside the worker process via `/collective_rpc` (requires `VLLM_SERVER_DEV_MODE=1`). This is how every step below
  is installed without reloading the model. See the security note in the README.
- It also exposes `exp_swap_build`, which can swap in `short_conv_attn_exp.py` (vLLM PR #55054, async H2D in the PLE
  short-conv metadata builder). That was measured as neutral on GB10 (E05) and is **not** activated by the installer.

---

## B. Hot-install steps (`scripts/install_stack.sh`, in order)

Each step writes a control file `overlays/runtime/<name>.txt` and runs the matching `exec_*.py`. Later steps rely on
objects created by earlier ones, so the order matters.

| # | Step (control) | Item | Effect | Measurement | Class |
|---|---|---|---|---|---|
| 1 | `exec_install_best.py` | INT4 coarse target head (E09/E12, superseded by step 9) + **W8 g64 HC mixers** (E13) | HC: 55.1 → 52.0 ms/step, +5.4% | sequential | head: exact up to ties; HC: quality-gated |
| 2 | `draft_w8` install | Draft dense Linears W8A16 g64 (E18) | draft pass 540 → 186 µs; +0.7% | microbench + sequential | output-exact (draft-only) |
| 3 | `ple_async` on | Async PLE handshake (E20) | GPU idle 2.45 → 0.55 ms; net noise-level | sequential | exact |
| 4 | `w6` on, then `freemx` | **W6A16 g32 target dense projections** (E22/E22d) | 51.52 → 49.27 ms/step, +4.3%; frees 2.76 GB | sequential | quality-gated |
| 5 | `dsize` on | Exact-size draft decode graphs 1..4 (E26a) | propose 5.32 → 4.64 ms; 49.64 → 48.39 ms/step | sequential | output-exact (draft-only) |
| 6 | `dlast` on | Draft-prefill MLP on last-token rows only (E26b) | propose 4.64 → 4.38 ms | sequential | output-exact (draft-only) |
| 7 | `moe_tactic` "16 49" | FlashInfer fused-MoE gemm1 tactic at M=4 (E30) | 48 layers 24.67 → 23.60 ms; gpu_verify −0.4 to −0.7 ms | microbench + alternating | bitwise identical |
| 8 | `draft_head_mode` int4r | Draft head INT4 coarse + FP32 refine top-8 (E32, superseded by step 10) | −0.83 ms/step | alternating | exact up to ties |
| 9 | `head2` "on 256" | **Target head INT2 coarse + exact BF16 refine** (E33) | −0.94 ms/step | alternating | exact up to ties (greedy) |
| 10 | `draft2` "int2r 64" | **Draft head INT2 coarse + FP32 refine top-64** (E34) | −0.70 ms/step | alternating | exact up to ties (draft-only) |
| 11 | `ple_zc` "on shm" | **PLE zero-copy rows + shared-memory request slot** (E35/E36) | −1.03 ms, then −0.57 ms/step | alternating | exact (rows bit-identical) |
| 12 | `ba`, `gate` on | Triton BF16 GEMV for GDN `in_proj_ba` and router gate (E37) | −0.36 ms/step | alternating | quality-gated (same precision) |
| 13 | `hc_red` on | Fused HC split-K reduce + cast (E38) | −0.44 ms/step | alternating | bitwise identical |
| 14 | `head_gate` "on 512" | Exactness gate for the fast target head (E39) | greedy unchanged; sampled uses exact head | gate counters | greedy: exact up to ties; everything else: exact |
| 15 | `probdraft` "on 20 0.9 0.9" | **Probabilistic fast-head draft for sampled requests** (E41) | sampled PROSE2 T1.0: ≈+14.7% (chained); PROSE T0.7 +5.0% | alternating | distribution-exact |
| 16 | `topkp` on | **Small-batch top-k/top-p fast path** (E42/U30a) | sampled −0.91 to −0.96 ms/step | alternating | bit-identical incl. tie order |

(Rows follow the command order in `install_stack.sh`; row 4 is two commands (`on`, then `freemx`) and row 12 is two consecutive commands (`ba`, `gate`).)

### 1. `exec_install_best.py` — INT4 target head (superseded) + W8 HC mixers

- **Target head, INT4 coarse + exact refine (E09/E12).** The target LM head (248,320 × 2,560 BF16 = 1.27 GB) was a
  5.2–5.3 ms eager GEMV per step. For decode-sized batches (≤16 rows) a coarse INT4-g32 head (0.36 GB) ranks all tokens;
  the top C=256 are recomputed exactly in BF16 and scattered in; all other logits are floored at the lowest refined
  value so an unrefined token can never outrank a refined one. Head call 5.21 → 1.85 ms; ordinary prose +6.1% vs the
  stock head on the same load (sequential). Verified on 6,466 rows (0 argmax mismatches) and 39,150 rows (2 mismatches,
  both exact BF16 ties between two kernels). **Superseded by step 9**, which keeps the same refine but uses an INT2 coarse
  pass. The step still runs because it saves the original head (`_exp_orig_compute_logits`) and creates the INT4 copy
  that step 9 reads; per the E33 record the INT4 copy (0.36 GB) stays resident.
- **HC mixers W8A16 group-64 (E13).** The hyper-connection mixers (97 target + 3 draft modules, 1.3 GB BF16) are
  bandwidth-bound GEMVs. They are replaced by a Triton W8A16 kernel (int8 symmetric, group 64 along K, fp32 scales,
  split-K 4 for the 336 × 10240 down/inject shape); BF16 weights are freed. Per layer at M=4: down 34.8 → 20.2 µs,
  up 28.4 → 16.2 µs. End to end 55.1 → 52.0 ms/step, ordinary prose 41.84 → 44.11 (+5.4%) (sequential).
  **Quality:** PPL 2.404 / 2.387 / 2.389 vs BF16 2.383–2.400; INT8 g64 fake-quant screen 2.391 / 2.382 / 2.389.
  Quality-gated numeric change.

### 2. Draft dense Linears W8A16 g64 (E18) — `exec_draft_w8.py`

- The MTP drafter's BF16 dense layers (`fc_embedding`, `fc_hidden`, attention qkv/o/indexer, shared expert, router gate;
  142 MB) move to the same W8 GEMV (relerr 0.7–1.7%). Per draft pass 540 → 186 µs. Ordinary prose 44.11 → 44.41 (+0.7%,
  sequential). Draft-only, so output-exact.

### 3. Async PLE handshake (E20) — `exec_ple_async.py`

- GB10 does not support CUDA stream memory operations, so the upstream semaphore path is unavailable and the host
  blocked on the PLE worker every step. Here the done flag is `cudaHostRegister`ed and a one-CTA Triton kernel spins on it
  inside the verify graph just before the PLE placeholder; the host returns immediately. GPU idle 2.45 → 0.55 ms, but the
  wait moved into the graph, so the net effect was noise-level (+0.8%, sequential). PPL 2.386. Kept because it is exact and
  **step 11 builds on it** (the zero-copy kernel reuses its flag registration).

### 4. Target dense projections W6A16 g32 (E22, E22d) — `exec_w6_install.py`

- The checkpoint stores the GDN `in_proj_qkvz`/`out_proj` and QSA `qkv`/`o` projections (96 matrices) in MXFP8. They are
  re-encoded from the MXFP8 weights to 6-bit RTN, group 32, fp16 scales, stored as a 4-bit plane plus a 2-bit plane
  (0.8125 B/param vs 1.03), with a Triton W6A16 GEMV (kernel relerr 0.2% vs dequant). Per layer at M=4: qkvz 189 → 154 µs,
  out 81 → 62, attn qkv 186 → 125, o 76 → 62.
- The `freemx` pass then frees the MXFP8 copies and routes prefill through W6 too: frees 2.76 GB (MemAvailable
  10.7 → 14.4 GB); prefill GEMMs are within 0–18% of MXFP8. `w6_shared` stays 0 (W6 on the shared expert was no gain, E22b).
- **Effect.** 51.52 → 49.27 ms/step, ordinary prose 44.78 → 46.69 (+4.3%) (sequential).
- **Quality (the W6 kernel path also used for teacher-forced scoring):** PPL 2.394 / 2.385 / 2.381 / 2.389 (band
  2.383–2.400); top-1 agreement 93.91% vs the 93.66% identical-run floor; functional gates pass; needle correct at 108k and
  201,650 tokens. INT5 (PPL mean 2.405, agreement below floor) and INT4 (2.541) were rejected. Quality-gated numeric change.

### 5–6. Draft graph padding removal (E26a/E26b) — `exec_dsize.py`, `exec_dlast.py`

- **E26a.** Draft decode graphs were captured only at 4 rows, so at S=1 draft steps 2–3 ran the MoE on 3 padding rows.
  Capturing sizes 1–4: propose 5.32 → 4.64 ms; 49.64 → 48.39 ms/step; ordinary 45.96 → 47.50 (sequential). PPL unchanged.
- **E26b.** In draft prefill only the last-token rows' MLP output is consumed (attention KV is written before the MLP), so
  the MLP runs only on those rows. Propose 4.64 → 4.38 ms; tok/s tie within noise (sequential).
- Both output-exact (draft-only). E26b changes draft numerics (M=1 vs M=4 Marlin), so near-tie drafts can flip.

### 7. MoE tactic override (E30) — `exec_moe_tactic_set.py`

- FlashInfer's CUTLASS fused-MoE autotuner chose gemm1 tactic 19 / gemm2 49 for M=4 on synthetic routing at warmup. A
  sweep on 48 real MoE layers found gemm1=16 **bitwise identical** and faster: 24.67 → 23.60 ms per 48 layers. In-server
  gpu_verify: [16, 49] 39.90 / 40.21 ms vs original 40.63 / 40.42 ms (alternating). Faster gemm2 tactics (56/59, −0.5 ms)
  were rejected because they are not bit-exact (relerr 5e-3).

### 8. Draft head INT4 + FP32 refine top-8 (E32, superseded) — `exec_draft_int4.py`

- INT4 coarse over the 64,877 FP8 draft rows, then an FP32 recompute of the top-8 from the FP8 rows. Head 667–686 →
  474–477 µs per draft step; −0.83 ms/step (alternating, 47.46 vs 48.29). Of 2,136 rows, 27 argmax flips, all exact ties.
- **Superseded by step 10.** It still runs because step 10 saves and depends on the function objects it installs (the
  original FP8 path and the E32 path, which step 10 can restore with mode `int4r`).

### 9. Target head INT2 coarse + exact BF16 refine (E33) — `exec_head2.py`

- Same design as step 1 with a 2-bit coarse pass: 4 levels × fp16 scale per 16 (238 MB). In a containment study the exact
  top-1 was always within coarse rank 17 (far inside C=256). Coarse GEMV at M=4 1,573 → 1,023 µs; full head 1.78 →
  1.20–1.29 ms; −0.94 ms/step (alternating, 47.01 vs 47.95). 0 argmax differences on 5,066 verification rows; PPL 2.391.
- Contract: **greedy argmax exact up to exact BF16 ties**. It is *not* distribution-exact (the refine differs from the
  reference GEMM by 1 ulp on 0.05% of logits, and top-20 containment fails on some rows), so step 14 restricts it to
  greedy batches and rebuilds it with C=512.

### 10. Draft head INT2 coarse + FP32 refine top-64 (E34) — `exec_draft_int2.py`

- INT2 coarse over the 64,877 draft rows (≈250 µs at M=1), FP32 recompute of the top-64 from the FP8 rows. Head
  473–498 → 311–319 µs; −0.70 ms/step (alternating, 46.53 vs 47.23). K=64 gave 0 "real" flips (logit gap > 1e-3) on
  2,328 rows. Exact up to ties, draft-only.

### 11. PLE zero-copy delivery + shared-memory request slot (E35, E36) — `exec_ple_zc.py` "on shm"

- **E35.** The PLE worker (B4) writes rows into a host-mapped shared buffer; inside the graph one Triton kernel spins on the
  done flag with acquire semantics and copies the rows to the GPU PLE buffer. No CPU-side CUDA copy or sync.
  −1.03 ms/step (alternating, 45.43 vs 46.46). 898 verified steps, 0 row mismatches; PPL 2.380 / 2.392.
- **E36.** The request is published through a shared-memory slot that the worker spin-polls (zmq kept as a deduplicated
  fallback): launch→receive 0.69 → 0.25 ms median; −0.57 ms/step (alternating, 44.93 vs 45.50). 846 verified steps, 0
  mismatches; PPL 2.371. After E36 the in-graph PLE wait is ≈6 µs per step, i.e. fully hidden.
- Cost: the worker busy-polls (≈10% of one core idle, ~20% under light load) and backs off after 5 ms without work.

### 12. Triton BF16 GEMVs for `in_proj_ba` and the router gate (E37) — `exec_ba_gemv.py`, `exec_gate_install.py`

- The last BF16 cuBLAS GEMMs in the decode graph (GDN `in_proj_ba` [96, 2560] × 36, router gate [512, 2560] × 48) go to a
  tuned Triton BF16 GEMV with fp32 accumulation (same weights, same precision), removing the cuBLAS split-K reduce kernels.
  −0.36 ms/step (alternating 5×/5×, 44.80 vs 45.16).
- Not bitwise identical: the router's top-10 expert set differs from cuBLAS on 3.9% of rows (near-ties), but against an
  FP32 reference Triton is slightly *closer* than cuBLAS (13,591 vs 13,772 differing rows). PPL 2.386; gates pass.
- The `in_proj_ba` autotuner rejects split-K/block combinations that do not tile K and any config with relerr ≥ 1e-2
  (fix B8; an invalid config once won the timing race after a relaunch and produced garbage).

### 13. Fused HC split-K reduce + cast (E38) — `exec_hc_red.py`

- The W8 HC down/inject GEMV epilogue (two ATen kernels: sum over split-K, cast to BF16) becomes one Triton kernel with the
  same summation order: **bitwise identical**. Removes 97 dependent graph nodes per verify step. −0.44 ms/step
  (alternating, 44.62 vs 45.06).

### 14. Exactness gate for the fast target head (E39) — `exec_head_gate.py` "on 512"

- The fast INT2 head is used **only** inside the target sampler for batches in which every request is greedy
  (temperature 0) and has no penalties, logit bias, allowed-token lists, bad words, thinking budget, logprobs, logprob
  token ids or grammar bitmask. Every other call (any sampled request, logprobs, guided/JSON decoding, prompt logprobs,
  dummy runs) uses the original BF16 head.
- The gate rebuilds the fast head with **C=512**: on 3,792 sampled-context rows C=256 had one genuine greedy error (a flat
  distribution whose true argmax fell outside the candidates); C=512 had 0 argmax differences, for +15 µs.
- Effect: greedy speed unchanged; sampled requests pay ≈+5 ms/step for exactness (the full head). Gates pass.

### 15. Probabilistic fast-head draft for sampled requests (E41) — `exec_probdraft.py` "on 20 0.9 0.9"

- With a greedy (one-hot) draft, a sampled request accepts a draft token with probability p(argmax); a probabilistic
  draft is accepted with Σ min(p, q). The speculator's draft sampler is overridden to use the fast reduced-vocab draft head
  (step 10), keep its top-20 logits, apply a draft-side top-p 0.9 and temperature × 0.9, and scatter them into a
  full-vocab BF16 row (−inf elsewhere). vLLM's Gumbel-coupled sampler caches exactly these logits as *q* for the
  rejection sampler. Greedy requests keep argmax drafting.
- **Effect (alternating A/B, same process).** PROSE2 T1.0/p0.95/k20: +10.8% for the top-20 draft (S2) and +3.56% more for
  the top-p/temperature shaping (S3), **≈+14.7% chained**. PROSE T0.7: +5.0% (6/6 prompts). S=2 T0.7 aggregate: +7.2%.
  150k-context decode: T0.7 +2.1%, T1.0 +4.9%. Code/JSON sampled: neutral. Greedy PROSE2: 54.37 → 54.37.
  Cost ≈0.2 ms/step.
- **Class.** distribution-exact (standard speculative sampling with the cached *q*). Rescoring sampled texts under the
  exact target head was within noise (T1.0: +0.054 ± 0.104 SE mean log-prob per token).

### 16. Small-batch top-k/top-p fast path (E42 / U30a) — `exec_topkp.py`

- In the pinned engine, sampler batches of fewer than 8 rows apply top-k/top-p with a full sort over the 248k vocabulary
  (1.02 ms per call; two calls per sampled step). When every row has 1 ≤ top_k ≤ 64, the hook instead does `topk(64)`,
  reorders to the stock stable-sort tie order, applies the k-th-value threshold (ties kept) and top-p over the survivors,
  and scatters into a −inf row: 0.27 ms. Other requests use the original function.
- **Exactness.** 0 of 2,048 real sampled rows (k20/p0.95 at T0.7 and T1.0) differ from the stock path, including tie
  order. Greedy batches never reach the hook.
- **Effect (alternating A/B, same process).** PROSE2 T1.0: 49.92 → 48.96 ms/step (−0.96 ± 0.13, 12/12 prompts faster);
  fresh PROSE3 T0.7: 49.73 → 48.82 ms/step (−0.91 ± 0.08, 8/8). ≈+1.9% at constant tok/step for sampled requests.

---

## Superseded steps that remain in the installer

| Step | Superseded by | Why it still runs |
|---|---|---|
| INT4 target head in `exec_install_best.py` (E09/E12) | E33 INT2 head (step 9), then gated by E39 | Saves the original BF16 head that E33/E39 fall back to, and builds the INT4 copy that `exec_head2.py` reads. |
| INT4 draft head `exec_draft_int4.py` (E32) | E34 INT2 draft head (step 10) | Saves the original FP8 draft-head function and installs the E32 function; `exec_draft_int2.py` requires both. |

## Items measured but not active

Neutral or rejected items are in [RESEARCH_LOG.md](RESEARCH_LOG.md). One approximate item was measured and is
deliberately **not** enabled: the S6 fast target head for sampled requests (see [DECISIONS.md](DECISIONS.md)).
