#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# Generic same-process on/off A/B for a live toggle exec. usage: toggle_ab.sh <outprefix> <exec.py> <ctlfile> <suite> <blocks> [q38bench args...]
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
PORT=${Q38_PORT:-5810}
OUT=$1; EX=$2; CTL=$3; SUITE=$4; BLOCKS=$5; shift 5
tog() { echo "$1" > overlays/runtime/$CTL; timeout 120 python3 bench/rpc.py exp_exec /exp/$EX | python3 -c 'import json,sys;print(json.load(sys.stdin)["results"][0].splitlines()[-1])'; }
ORDER=(off on on off off on on off)
ACMD=${ACMD:-off}; BCMD=${BCMD:-on}; ATAG=${ATAG:-off}; BTAG=${BTAG:-on}
for ((b=0; b<BLOCKS; b++)); do
  arm=${ORDER[$b]}; if [ $arm = off ]; then tog "$ACMD"; tag=$ATAG; else tog "$BCMD"; tag=$BTAG; fi
  python3 bench/q38bench.py --port $PORT --tag $tag --suite $SUITE --streams 1 --repeats 1 --out ${OUT}.jsonl --texts ${OUT}_texts.jsonl --note "block$b" "$@" | tail -n +2
done
# restore the promoted setting (FINAL, default = BCMD) so an A/B never leaves the stack degraded
tog "${FINAL:-$BCMD}"; echo AB_DONE
