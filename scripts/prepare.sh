#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# prepare.sh — one-time host-side preparation (no model download):
#   1. build the packed PLE table (~27 GiB under $Q38_PLE_CACHE) with MiaAI-Lab's builder, if missing;
#   2. compile the 1-line ARM memory-barrier helper used by the PLE zero-copy path (overlays/runtime/exp_dmb.c);
#   3. build the r32a MTP drafter patch from the committed LoRA (scripts/build_adapter.sh), if missing;
#   4. generate the E44 prefix-cache block-drop backport (MiaAI-Lab PR #71 = vllm#53388) from the image's own files.
set -euo pipefail
. "$(dirname "$0")/common.sh"
[[ -f $SNAPDIR/model.safetensors.index.json ]] || die "checkpoint not found at $SNAPDIR — run ./download.sh first"
if ls "$PLEDIR"/*.packed_u8 >/dev/null 2>&1; then ok "packed PLE table present ($PLEDIR)"
else
  [[ "${Q38_DRY_RUN:-0}" == 1 ]] && { info "DRY RUN: would build packed PLE table into $PLEDIR (~27 GiB, ~40 s, CPU only)"; }
  if [[ "${Q38_DRY_RUN:-0}" != 1 ]]; then
    info "building packed PLE table (one-time, ~27 GiB written to $PLEDIR, CPU only)"
    mkdir -p "$PLEDIR"
    docker run --rm --label q38fn-opt=1 --memory 6g --cpus 8 --network none \
      -v "$HFH/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4:/m:ro" -v "$PLEC:/out" \
      -v "$REPO/files/mia/build_ple_packed_table.py:/b.py:ro" \
      --entrypoint python3 "$IMAGE" -u /b.py "/m/snapshots/$MODEL_REVISION" "/out/Mia-AiLab--Qwen3.8-Flash-Next-NVFP4"
    ok "packed PLE table built"
  fi
fi
if [[ -f $PATCH && $(sha "$PATCH") == "$PATCH_SHA" ]]; then ok "r32a drafter patch present and verified"
elif [[ "${Q38_DRY_RUN:-0}" == 1 ]]; then info "DRY RUN: would build $PATCH from adapters/r32a/lora_r32a.pt"
else bash "$REPO/scripts/build_adapter.sh"; fi
SO=$REPO/overlays/runtime/exp_dmb.so
if [[ -f $SO ]]; then ok "exp_dmb.so present"
elif [[ "${Q38_DRY_RUN:-0}" == 1 ]]; then info "DRY RUN: would compile overlays/runtime/exp_dmb.c"
else
  if command -v gcc >/dev/null; then gcc -O2 -shared -fPIC -o "$SO" "$REPO/overlays/runtime/exp_dmb.c"
  else docker run --rm --network none -u "$(id -u):$(id -g)" -v "$REPO/overlays/runtime:/w" --entrypoint gcc "$IMAGE" -O2 -shared -fPIC -o /w/exp_dmb.so /w/exp_dmb.c; fi
  ok "compiled exp_dmb.so"
fi
BD=$REPO/overlays/block_drop
if [[ -f $BD/out/config/speculative.py ]] && grep -q disable_eagle_block_drop "$BD/out/config/speculative.py"; then ok "block-drop backport present"
elif [[ "${Q38_DRY_RUN:-0}" == 1 ]]; then info "DRY RUN: would generate overlays/block_drop/out from the image (MiaAI-Lab PR #71)"
else
  docker run --rm --network none -u "$(id -u):$(id -g)" -v "$BD:/w" --entrypoint bash "$IMAGE" -c '
    set -e; P=/usr/local/lib/python3.12/dist-packages/vllm; cd /w
    for f in $(python3 patch_block_drop.py --list); do mkdir -p orig/$(dirname $f); cp $P/$f orig/$f; done
    python3 patch_block_drop.py /w/orig /w/out'
  ok "generated block-drop backport (6 files)"
fi
QF=$REPO/overlays/runtime/qf
if [[ -f $QF/build/qf_moe.so ]]; then ok "qf_moe.so present"
elif [[ "${Q38_DRY_RUN:-0}" == 1 ]]; then info "DRY RUN: would build overlays/runtime/qf/build/qf_moe.so (SSHdotCodes NVFP4 decode MoE, sm_121a)"
else
  docker run --rm --network none --gpus all -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$QF:/qf" --entrypoint python3 "$IMAGE" /qf/build.py
  ok "built qf_moe.so"
fi
