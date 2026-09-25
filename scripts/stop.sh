#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# stop.sh — stop the server container started by serve.sh and its memory watchdog; keep a bounded log tail.
set -uo pipefail
. "$(dirname "$0")/common.sh"; EXP=${Q38_STATE:-$REPO/.state}
pkill -f "memwatch.sh $NAME( |$)" 2>/dev/null || true
if docker ps -a --format '{{.Names}}' | grep -q "^$NAME\$"; then
  mkdir -p $EXP/logs; TAG=$(cat $EXP/logs/CURRENT_TAG 2>/dev/null || echo unknown)
  docker logs --tail 1500 $NAME > $EXP/logs/${TAG}-final-$(date +%Y%m%dT%H%M%S)-container.log 2>&1 || true
  docker stop -t 30 $NAME >/dev/null 2>&1 || true; docker rm -f $NAME >/dev/null 2>&1 || true
  echo "stopped $NAME"
else echo "$NAME not present"; fi
