#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# deploy/llama-swap-start.sh <port> — llama-swap `cmd` adapter for the stable E46 stack.
#   Launches vLLM on an internal port (<port>+20000, 127.0.0.1), hot-installs E46, and only then opens <port> (socat), so
#   llama-swap's checkEndpoint cannot succeed before the stack is installed. Stays in the foreground until the container exits.
# Pair with `cmdStop: <repo>/deploy/llama-swap-stop.sh`. Assumes ./run.sh (or scripts/prepare.sh) ran once already.
set -Eeuo pipefail
port=${1:?llama-swap backend port required}
REPO=$(cd "$(dirname "$0")/.." && pwd)
export Q38_STATE=${Q38_STATE:-$REPO/.state}
export Q38_SERVED_NAME=${Q38_SERVED_NAME:-qwen3.8-flash-next-optimized}   # must equal the llama-swap alias (requests are proxied verbatim)
mkdir -p "$Q38_STATE/logs"
exec >>"$Q38_STATE/logs/llama-swap-start.log" 2>&1
echo "=== $(date -Is) llama-swap start port=$port"
. "$REPO/scripts/common.sh"
# The vLLM server listens on an internal port; llama-swap's <port> is only opened (by socat) after the E46 install
# finished, so llama-swap never routes a request to a half-installed server.
inner=$(( port + 20000 ))
export Q38_PORT=$inner
bash "$REPO/scripts/serve.sh"
echo "=== $(date -Is) E46 installed; exposing 127.0.0.1:$port -> $inner"
socat TCP-LISTEN:"$port",bind=127.0.0.1,reuseaddr,fork TCP:127.0.0.1:"$inner" &
sp=$!
trap 'kill $sp 2>/dev/null; bash "$REPO/scripts/stop.sh"' TERM INT
docker wait "$NAME" &
wait -n
kill $sp 2>/dev/null || true
