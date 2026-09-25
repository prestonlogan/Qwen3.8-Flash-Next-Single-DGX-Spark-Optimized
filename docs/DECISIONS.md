# Open decisions

Items that are measured but deliberately **not** in the default configuration, because adopting them is a quality-risk or policy decision rather than an engineering one.

## D-S6 — Approximate fast target head for sampled requests (not enabled)

| | |
|---|---|
| What | For sampled requests, compute the target logits with the INT2 coarse head, then compute exact BF16 logits only for its top-C candidates (C=1024–2048). Every other token gets −inf. |
| Gain | Direct same-process A/B on PROSE2 (dev set), T1.0 / top-p 0.95 / top-k 20: **48.38 → 52.35 tok/s (+8.2%)**, ms/step 49.02 → 45.12. |
| Evidence of fidelity | 0 changed rows (total-variation distance 0 between processed distributions) at C ≥ 1024. This covers 4,096 captured PROSE2 rows at T0.7 and T1.0, plus 4,096 rows from code, JSON and other prose. At C=512, 1–2 rows changed; at C=256, 7. |
| Why not default | The shortcut changes the **target** distribution path. Unlike draft-side changes, rejection sampling does not correct it. Exactness cannot be certified: Cauchy–Schwarz and low-rank-correction bounds leave thousands of candidates (S7). Zero observed changes is empirical evidence, not a guarantee. It was validated only for top_k=20, top_p 0.95 and T ≤ 1. Without top-k truncation, untruncated T1.0 rows showed TV up to 0.26. |
| If adopted | Gate it strictly to that validated setting: top_k==20, top_p 0.95, T ≤ 1, no logprobs, penalties or grammar, C=2048. Use the exact head for everything else. Confirm on a fresh disjoint set before promotion. |

## D-PLE — upstream recipe PLE fix (informational)

This repository always uses the PLE row-delivery fix (`overlays/runtime/ple_layer_fixrows.py`). The Mia recipe version we started from delivered only token row 0 through the CPU-offload PLE path (PPL 4.998 vs 2.382). If you run the original recipe elsewhere, use a version that includes an equivalent fix; MiaAI Lab's later branch ships one.
