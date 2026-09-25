# Changelog

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
