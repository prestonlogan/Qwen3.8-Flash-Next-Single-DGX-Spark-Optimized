#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# start.sh — launch the optimized server (assumes ./download.sh and scripts/prepare.sh have run). See ./run.sh for the guided path.
exec bash "$(dirname "$0")/scripts/serve.sh" "$@"
