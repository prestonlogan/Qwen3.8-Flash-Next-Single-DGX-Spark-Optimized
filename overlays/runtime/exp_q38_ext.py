# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
"""Worker extension for the isolated decode-speed experiment (collective_rpc hooks)."""
import torch


class ExpWorkerExt:
    def exp_set_draft_mask(self, path: str) -> str:
        m = self.model_runner.speculator.model
        bias = getattr(m, "_exp_draft_bias", None)
        if bias is None:
            return "no mask buffer"
        tgt = m._draft_id_to_target_id.long()
        with open(path) as f:
            ids = sorted({int(l) for l in f if l.strip()})
        ids_t = torch.tensor(ids, dtype=torch.long, device=tgt.device)
        active = torch.isin(tgt, ids_t)
        new = torch.where(active, torch.zeros_like(bias), torch.full_like(bias, float("-inf")))
        torch.cuda.synchronize()
        bias.copy_(new)
        torch.cuda.synchronize()
        return f"active {int(active.sum())} of {tgt.numel()} (file ids {len(ids)})"

    def exp_info(self) -> str:
        m = self.model_runner.speculator.model
        out = {k: tuple(getattr(m, k).shape) for k in
               ("_draft_lm_head_weight", "_draft_lm_head_fp8", "_exp_draft_bias", "_draft_id_to_target_id")
               if getattr(m, k, None) is not None}
        return str(out)


def _swap_build(which: str) -> str:
    import importlib.util, types
    import vllm.v1.attention.backends.short_conv_attn as orig_mod
    cls = orig_mod.PleShortConvAttentionMetadataBuilder
    if not hasattr(cls, "_exp_orig_build"):
        cls._exp_orig_build = cls.build
    if which == "orig":
        cls.build = cls._exp_orig_build
        return "build=orig"
    spec = importlib.util.spec_from_file_location("_exp_sca", "/exp/short_conv_attn_exp.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    src_fn = m.PleShortConvAttentionMetadataBuilder.build
    o = cls._exp_orig_build
    if src_fn.__code__.co_freevars != o.__code__.co_freevars:
        return f"freevar mismatch {src_fn.__code__.co_freevars} vs {o.__code__.co_freevars}"
    fn = types.FunctionType(src_fn.__code__, orig_mod.__dict__, "build", src_fn.__defaults__, o.__closure__)
    fn.__kwdefaults__ = src_fn.__kwdefaults__
    cls.build = fn
    return "build=exp(pr55054)"


def _exp_swap_build(self, which: str) -> str:
    return _swap_build(which)


ExpWorkerExt.exp_swap_build = _exp_swap_build


def _exp_exec(self, path: str) -> str:
    """Run an experiment script inside the worker process (hot-swap without model reload).
    The script sees `worker` and must set `RESULT` (str)."""
    g = {"worker": self, "__name__": "exp_exec"}
    with open(path) as f:
        exec(compile(f.read(), path, "exec"), g)
    return str(g.get("RESULT", "ok"))


ExpWorkerExt.exp_exec = _exp_exec
