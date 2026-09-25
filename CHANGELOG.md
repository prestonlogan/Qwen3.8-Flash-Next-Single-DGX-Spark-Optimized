# Changelog

## v0.5.0 — 2026-09-25 — E46 (sampled decode: lossless compressed exact head)

- **Exact target head for sampled requests** (default chat T1.0 / p.95; greedy already uses the E39 fast head).
  - The BF16 lm_head is stored losslessly: a sign+mantissa byte, a 4-bit exponent offset from the per-64 group maximum, and a sparse fp32 add-back for the 0.01% escapes. That is 0.96 GB instead of 1.27 GB; weights reconstruct bit-exactly.
  - A Triton GEMV reads the compressed form: 5.47 → 4.04 ms at M=4.
  - Logits: 99.98% bitwise-equal to cuBLAS, the rest within 1 bf16 ulp (accumulation order only); TV ≤ 2.4e-6 at T1.0.
- **Measured (same process, sampled):** ms/step −1.42 ± 0.22 (PROSE2, 12/12 prompts) and −1.58 ± 0.37 (PROSE3, 8/8), i.e. ≈ −3%. Greedy unchanged.
- **Memory:** install-only INT4 head copies are now freed, so the net is ≈ +0.5 GB vs E45. Idle MemAvailable on a fresh boot is 14.0 GB.
- `Q38_HX=0` restores the cuBLAS BF16 exact head; `Q38_FREE=0` keeps the INT4 copies.
- **Concurrency verified:** warmed dual 150k+150k runs at ≈40.2 tok/s per stream (≈80.5 aggregate, greedy), single warm 150k at 52.1 tok/s; minimum host MemAvailable 13.3 GB.

## v0.4.0 — 2026-09-25 — E45 (decode: faster NVFP4 routed-expert kernel)

- **Decode MoE kernel.** The routed experts of the 1–8-row verify/draft target graphs use SSHdotCodes' `qf_moe.cu` ([qwen-3.8-flash-next-pro6000](https://github.com/SSHdotCodes/qwen-3.8-flash-next-pro6000) @ `8e5be44`, Apache-2.0).
  - It is a 2-kernel dp4a W4A4 MoE that reads each selected expert once, with the same quantization recipe as FlashInfer CUTLASS. Built for sm_121a.
  - Two-line header change only (c10 stream headers). See `THIRD_PARTY.md`.
  - Other shapes fall back to FlashInfer CUTLASS. `Q38_QF=0` restores E44.
- **Measured:** routed MoE ≈203 → 231–245 GB/s; verify graph 38.5 → 37.0 ms.
  - Speed: PROSE3 greedy +4.0%, default-chat T1.0 +2.8–3.6%, 150k greedy ≈+6–8% preliminary (n=2 per arm, cold/warm imbalance).
  - Quality: PPL 2.378 → 2.382 (noise band 2.37–2.40); tool, JSON, reasoning and code gates pass; logprob divergence within same-arm noise.
  - Not bitwise identical to CUTLASS (different summation order).

## v0.3.0 — 2026-09-25 — E44 (responsiveness / TTFT; decode tok/s unchanged)

- **Keep the trailing prefix-cache block.** With MTP, the pinned vLLM drops the last scheduler block from every prefix-cache hit. On this hybrid model that block is 1,664 tokens, so each warm turn re-prefills about 1.7k tokens that are already cached.
  - `overlays/block_drop/patch_block_drop.py` is MiaAI-Lab's backport of vllm#53388: [PR #71](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/pull/71), commit `ffc41629`, author usmaneth, merged via PR #72. It's copied unmodified.
  - `scripts/prepare.sh` generates the 6 patched files from the image. `serve.sh` mounts them and sets `disable_eagle_block_drop`. `Q38_BLOCK_DROP=1` restores stock behaviour.
- **Paired test:** 6 sessions, about 8.4k tokens of context; E43 vs E44, same conversation tokens.
  - Warm second turn TTFT: **1.24 → 0.68 s**.
  - Turn after a ~3k-word tool result: **3.51 → 2.65 s**.
  - Cold first turn: 5.21 → 5.18 s (unchanged).
  - Prefix-cache hits per warm turn: 6,656 → 8,320 of 8,409 prompt tokens.
  - Decode tok/s and tok/step unchanged.
  - Warm-vs-cold top-20 logprob divergence is the same as the E43 baseline (+0.032 vs +0.036 TV above repeat noise).

## v0.2.0 — 2026-09-25 — E43

- **Exactness fix for sampled requests.** The E41 probabilistic draft sampled its tokens with the Gumbel noise key that the pinned vLLM V2 rejection sampler later reuses for the residual resample of the same row. That coupling biases sampled outputs.
  - The draft now uses a disjoint key range (`probdraft on 20 0.9 0.9 indep`).
  - Speed-neutral: PROSE2 +0.9% ± 1.0, PROSE3 +0.4% ± 0.4.
  - Greedy is unaffected.
- **CK42 current-state benchmark**, all direct same-process measurements:
  - PROSE ordinary-5 greedy: 53.23 tok/s, +45% vs E01.
  - PROSE2 default chat: 48.57 tok/s vs 43.69 with E41/E42 off.
  - PPL 2.3787, all gates pass, dual 144k/150k long context OK.

## v0.1.0 — 2026-09-25 (initial private release)

Promoted configuration **E42**, built on the MiaAI-Lab single-DGX-Spark recipe (upstream commit `d038090`) and `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6`.

**Boot-time items:**

- PLE row-delivery correctness fix (E10/E11).
- FP8 draft head over a 64,877-token draft vocabulary (E03/E04).
- r32a MTP drafter LoRA (T8/E40), shipped as a 14.6 MB LoRA plus a deterministic merge (`scripts/build_adapter.sh`, sha256-verified output).
- PLE zero-copy worker (E35/E36).

**Hot-installed stack (`scripts/install_stack.sh`):**

- W8 HC mixers (E13), W8 draft dense layers (E18), async PLE handshake (E20), W6A16 target dense projections (E22).
- Exact-size draft graphs and last-row draft MLP (E26).
- MoE gemm1 tactic override (E30).
- INT2 coarse + exact-refine target head (E33) with the greedy exactness gate (E39).
- INT2 coarse + refine draft head (E34).
- PLE zero-copy + shared-memory request slot (E35/E36).
- Triton BF16 GEMVs for the router gate and in_proj_ba (E37).
- Fused HC reduce (E38).
- Probabilistic fast-head drafting for sampled requests (E41).
- Small-batch top-k/top-p fast path (E42).

**Tooling:**

- `run.sh` (one command, with `--dry-run`), explicit `download.sh`, `prepare.sh`, and the `bench/` suite.
- A-B toggling, tracked jobs, gates, and the PROSE3 confirmation set.

**Validation:**

- A clean-copy `./run.sh` built all artifacts, launched, installed E42 and passed health and all functional gates.
- Single-pass smoke: PROSE2 greedy 53.86 tok/s; PROSE3 T1.0 48.15 tok/s.
