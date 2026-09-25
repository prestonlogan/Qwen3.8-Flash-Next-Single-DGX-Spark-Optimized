# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# Shared defaults for the top-level scripts. Sourced, not executed. Values can be overridden in .env or the environment.
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[[ -f "$REPO/.env" ]] && set -a && . "$REPO/.env" && set +a
MODEL_ID=Mia-AiLab/Qwen3.8-Flash-Next-NVFP4
MODEL_REVISION=${Q38_MODEL_REVISION:-925d7be6c14c6c9442ef83e8f05b5a3c39304f69}
IMAGE=${Q38_IMAGE:-vllm/vllm-openai@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8}
HFH=${Q38_HF_HOME:-$HOME/.cache/huggingface}
PLEC=${Q38_PLE_CACHE:-$HOME/.cache/vllm/ple_cache}
SNAPDIR=$HFH/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/snapshots/$MODEL_REVISION
PLEDIR=$PLEC/Mia-AiLab--Qwen3.8-Flash-Next-NVFP4
PATCH=$REPO/adapters/r32a/mtp_patch_r32a.safetensors
PATCH_SHA=bd3c1807244253342c6c83e7aaa2864856456e4ffcae3baa069cbde36076c302
LORA_SHA=3e5a76538018b88648705fbda6420f3d4a4859f805d3b4b30dd5c9a77e4d9680
PORT=${Q38_PORT:-5810}
NAME=${Q38_NAME:-q38fn-opt}
info() { printf '\033[36m[..]\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[!!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[xx]\033[0m %s\n' "$*" >&2; exit 1; }
sha()  { sha256sum "$1" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$1" | cut -d' ' -f1; }
