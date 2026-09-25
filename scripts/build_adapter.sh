#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# build_adapter.sh — deterministically merge the committed r32a LoRA (14.6 MB) into the checkpoint's BF16 MTP tensors,
# producing adapters/r32a/mtp_patch_r32a.safetensors (175 MB, 22 tensors). CPU-only, network-less container, ~1 min.
# The output is verified against the recorded SHA-256 (bit-identical to the patch used for all measurements).
set -euo pipefail
. "$(dirname "$0")/common.sh"
L=$REPO/adapters/r32a/lora_r32a.pt
[[ $(sha "$L") == "$LORA_SHA" ]] || die "LoRA hash mismatch for $L"
[[ -f $SNAPDIR/model.safetensors.index.json ]] || die "checkpoint not found at $SNAPDIR"
T=$(mktemp -d); trap 'docker run --rm --network none -v "$T:/w" --entrypoint rm "$IMAGE" -rf /w/out >/dev/null 2>&1; rm -rf "$T"' EXIT
mkdir -p "$T/out"; cp "$L" "$T/out/lora_r32a.pt"
docker run --rm --label q38fn-opt=1 --network none --cpus 4 --memory 8g -e CUDA_VISIBLE_DEVICES= \
  -e Q38_SNAPSHOT=/root/.cache/huggingface/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/snapshots/$MODEL_REVISION/ \
  -v "$HFH:/root/.cache/huggingface:ro" -v "$T/out:/t8" -v "$REPO/overlays/runtime/t8_merge.py:/merge.py:ro" \
  --entrypoint bash "$IMAGE" -c "python3 /merge.py r32a 32 64 | tail -1 && chmod a+r /t8/mtp_patch_r32a.safetensors"
cp "$T/out/mtp_patch_r32a.safetensors" "$PATCH.tmp"
got=$(sha "$PATCH.tmp"); [[ $got == "$PATCH_SHA" ]] || { rm -f "$PATCH.tmp"; die "merged patch hash $got != expected $PATCH_SHA"; }
mv "$PATCH.tmp" "$PATCH"; ok "built $PATCH (sha256 verified)"
