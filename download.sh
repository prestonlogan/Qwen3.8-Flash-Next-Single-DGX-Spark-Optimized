#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# download.sh — EXPLICITLY download the pinned Mia-AiLab/Qwen3.8-Flash-Next-NVFP4 checkpoint (~100 GB) into the HF cache.
# Nothing else in this repo downloads the model. Resumable. Uses the pinned image's huggingface_hub (no host Python deps).
#   ./download.sh            download (asks for confirmation)
#   ./download.sh --yes      no prompt
set -euo pipefail
. "$(dirname "$0")/scripts/common.sh"
if [[ -f $SNAPDIR/model.safetensors.index.json ]]; then ok "checkpoint already present: $SNAPDIR"; exit 0; fi
free=$(df -BG --output=avail "$HFH" 2>/dev/null | tail -1 | tr -dc 0-9 || echo 0)
info "will download $MODEL_ID@$MODEL_REVISION (~100 GB) into $HFH (free: ${free} GiB)"
(( free >= 110 )) || die "need >=110 GiB free in $HFH"
if [[ "${1:-}" != --yes ]]; then read -r -p "Proceed with ~100 GB download? [y/N] " a; [[ $a == y || $a == Y ]] || die "aborted"; fi
mkdir -p "$HFH"
docker run --rm -i --label q38fn-opt=1 -e HF_HOME=/hf ${HF_TOKEN:+-e HF_TOKEN} -v "$HFH:/hf" --entrypoint python3 "$IMAGE" - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download("$MODEL_ID", revision="$MODEL_REVISION"))
PY
[[ -f $SNAPDIR/model.safetensors.index.json ]] || die "download incomplete; rerun ./download.sh"
# serve.sh pins --revision, but also record refs/main for tools that resolve the bare repo id offline
R=$HFH/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/refs; [[ -f $R/main ]] || { mkdir -p "$R" && printf %s "$MODEL_REVISION" > "$R/main"; }
ok "checkpoint ready: $SNAPDIR"
