#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
"""Merge a T8 LoRA (+norm deltas) into BF16 MTP tensors; write only changed tensors (checkpoint names) to /t8/mtp_patch_<tag>.safetensors."""
import sys, json, glob, torch
from safetensors import safe_open
from safetensors.torch import save_file
tag = sys.argv[1]; rank = int(sys.argv[2]) if len(sys.argv) > 2 else 32; alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 64
import os
M = os.environ.get("Q38_SNAPSHOT") or glob.glob("/root/.cache/huggingface/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/snapshots/*/")[0]
wm = json.load(open(M + "model.safetensors.index.json"))["weight_map"]
sd = torch.load(f"/t8/lora_{tag}.pt"); out = {}; sc = alpha / rank
names = sorted(set(k.rsplit(".", 1)[0] for k in sd))
with safe_open(M + wm["mtp.fc_hidden.weight"], "pt") as f:
    for n in names:
        ck = "mtp." + n + ".weight"; w = f.get_tensor(ck).float()
        if n + ".full" in sd:
            w = sd[n + ".full"].float()
        if n + ".A" in sd:
            w = w + (sd[n + ".B"].float() @ sd[n + ".A"].float()) * sc
        if n + ".d" in sd:
            w = w + sd[n + ".d"].float()
        nw = w.to(torch.bfloat16); old = f.get_tensor(ck)
        out[ck] = nw.contiguous()
        print(ck, tuple(nw.shape), "rel_change %.4f" % ((nw.float() - old.float()).norm() / old.float().norm()).item())
save_file(out, f"/t8/mtp_patch_{tag}.safetensors", metadata={"t8": tag})
print("wrote", len(out), "tensors", sum(v.numel() * 2 for v in out.values()) / 1e6, "MB")
