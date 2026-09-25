# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
import json, os, sys, urllib.request
method = sys.argv[1]; args = sys.argv[2:]
req = urllib.request.Request(f"http://127.0.0.1:{os.environ.get('Q38_PORT', '5810')}/collective_rpc",
    data=json.dumps({"method": method, "args": args}).encode(), headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(req, timeout=900).read().decode())
