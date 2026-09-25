#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# run.sh — one-command path to the current recommended configuration on ONE DGX Spark.
#   ./run.sh              check → prepare (PLE table, drafter patch) → launch → install stack → health check
#   ./run.sh --dry-run    run every check and print what would happen; no builds, no launch
# The ~100 GB checkpoint is never downloaded implicitly: run ./download.sh yourself first.
set -euo pipefail
. "$(dirname "$0")/scripts/common.sh"
DRY=0; [[ "${1:-}" == --dry-run || "${1:-}" == -n ]] && DRY=1
export Q38_DRY_RUN=$DRY
info "1/6 host checks"
[[ $(uname -m) == aarch64 ]] || warn "not aarch64 — this recipe targets DGX Spark (GB10)"
gpu=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)
[[ -n $gpu ]] || die "nvidia-smi not available"
[[ $gpu == *GB10* || $gpu == *"12.1"* ]] && ok "GPU: $gpu" || warn "GPU '$gpu' is not GB10/SM121 — untested"
command -v docker >/dev/null || die "docker not found"; docker info >/dev/null 2>&1 || die "docker daemon not reachable by $USER"
docker info 2>/dev/null | grep -qi nvidia || warn "NVIDIA container runtime not reported by docker info"
for c in python3 curl ss; do command -v $c >/dev/null || die "$c not found"; done
avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
(( avail >= 104 )) && ok "MemAvailable ${avail} GiB" || { [[ $DRY == 1 ]] && warn "MemAvailable ${avail} GiB < 104 needed (other servers running?)" || die "MemAvailable ${avail} GiB < 104 — stop other GPU/LLM servers first"; }
info "2/6 image"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then ok "image present"; else
  [[ $DRY == 1 ]] && info "DRY RUN: would docker pull $IMAGE" || docker pull "$IMAGE"; fi
info "3/6 checkpoint"
[[ -f $SNAPDIR/model.safetensors.index.json ]] && ok "checkpoint $MODEL_ID@${MODEL_REVISION:0:8}" || die "checkpoint missing at $SNAPDIR — run ./download.sh (explicit ~100 GB download)"
info "4/6 prepare (PLE table, r32a drafter patch)"
bash "$REPO/scripts/prepare.sh"
info "5/6 launch + install optimization stack"
if docker ps -a --format '{{.Names}}' | grep -q "^$NAME\$"; then die "container $NAME exists — ./stop.sh first"; fi
if ss -ltn | grep -q ":$PORT "; then die "port $PORT in use"; fi
if [[ $DRY == 1 ]]; then info "DRY RUN: would run scripts/serve.sh (container $NAME, port $PORT) then scripts/install_stack.sh"; ok "dry run complete"; exit 0; fi
bash "$REPO/scripts/serve.sh"
info "6/6 health"
code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health || true); [[ $code == 200 ]] || die "health check failed ($code)"
ok "serving OpenAI-compatible API at http://127.0.0.1:$PORT/v1  (model: ${Q38_SERVED_NAME:-qwen3.8-flash-next})"
ok "active configuration: E43 (see docs/OPTIMIZATIONS.md); stop with ./stop.sh"
