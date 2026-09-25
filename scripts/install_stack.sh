#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# install_stack.sh — hot-install the promoted optimization stack (E42) into a running server started by serve.sh.
# Each step runs one overlays/runtime/exec_*.py inside the vLLM worker via collective_rpc (VLLM_SERVER_DEV_MODE).
# Control files (<name>.txt) are written next to the exec scripts; they are git-ignored runtime state.
# Order matters (later steps assume earlier ones). Total ~10-20 s. See docs/OPTIMIZATIONS.md for each item.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); O=$REPO/overlays/runtime
export Q38_PORT=${Q38_PORT:-5810}
run() { timeout 180 python3 "$REPO/bench/rpc.py" exp_exec "/exp/$1" | python3 -c 'import json,sys;print(json.load(sys.stdin)["results"][0])'; }
run exec_install_best.py                                        # INT4 coarse target head (superseded by E33 below) + W8 g64 HC mixers
echo install > $O/draft_w8.txt;   run exec_draft_w8.py          # E18 draft dense Linears W8A16 g64
echo on > $O/ple_async.txt;       run exec_ple_async.py         # E20 async PLE handshake (in-graph flag spin)
echo 0 > $O/w6_shared.txt
echo on > $O/w6.txt;              run exec_w6_install.py        # E22 W6A16 g32 target dense projections (quality-gated numeric change)
echo freemx > $O/w6.txt;          run exec_w6_install.py        # free the MXFP8 copies (W6 serves prefill too)
echo on > $O/dsize.txt;           run exec_dsize.py             # E26 exact-size draft decode graphs 1..4
echo on > $O/dlast.txt;           run exec_dlast.py             # E26b draft-prefill MLP on last-token rows only
echo "16 49" > $O/moe_tactic.txt; run exec_moe_tactic_set.py    # E30 FlashInfer fused-MoE gemm1 tactic at M=4 (bitwise identical)
echo 8 > $O/draft_head_k.txt; echo int4r > $O/draft_head_mode.txt; run exec_draft_int4.py   # E32 (superseded by E34)
echo "on 256" > $O/head2.txt;     run exec_head2.py             # E33 target head: INT2 coarse + exact BF16 refine (C=256)
echo "int2r 64" > $O/draft2.txt;  run exec_draft_int2.py        # E34 draft head: INT2 coarse + FP32 refine top-64
echo "on shm" > $O/ple_zc.txt;    run exec_ple_zc.py            # E35/E36 PLE zero-copy rows + shared-memory request slot
echo on > $O/ba.txt;              run exec_ba_gemv.py           # E37 GDN in_proj_ba Triton BF16 GEMV (validity + relerr gate)
echo on > $O/gate.txt;            run exec_gate_install.py      # E37 router gate Triton BF16 GEMV
echo on > $O/hc_red.txt;          run exec_hc_red.py            # E38 fused HC split-K reduce+cast (bitwise identical)
echo "on 512" > $O/head_gate.txt; run exec_head_gate.py         # E39 fast target head only for all-greedy plain batches; exact head otherwise
echo "on 20 0.9 0.9" > $O/probdraft.txt; run exec_probdraft.py  # E41 probabilistic fast-head draft for sampled requests (exact)
echo on > $O/topkp.txt;          run exec_topkp.py             # E42 small-batch top-k/top-p fast path (bit-identical to stock incl. tie order)
echo "INSTALL_OK"
