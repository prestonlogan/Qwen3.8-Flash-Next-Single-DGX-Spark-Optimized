#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# deploy/llama-swap-stop.sh — llama-swap `cmdStop` for deploy/llama-swap-start.sh (graceful container stop + watchdog).
REPO=$(cd "$(dirname "$0")/.." && pwd)
export Q38_STATE=${Q38_STATE:-$REPO/.state}
exec bash "$REPO/scripts/stop.sh"
