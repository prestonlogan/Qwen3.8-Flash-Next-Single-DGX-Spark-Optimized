# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# Router gate (BF16 [512,2560]) -> Triton BF16 GEMV for M<=16 (same weights/precision; fp32 accum). Mode /exp/gate.txt on|off
import importlib.util, torch, gc
s_ = importlib.util.spec_from_file_location("exp_bf16gemv", "/exp/exp_bf16gemv.py"); BG = importlib.util.module_from_spec(s_); s_.loader.exec_module(BG)
r = worker.model_runner; model = r.model
mode = open("/exp/gate.txt").read().split()[0]
gates = [m.mlp.gate for m in model.language_model.model.layers]
def recapture():
    for mg in [r.cudagraph_manager] + [getattr(r.speculator, k) for k in ("prefill_cudagraph_manager", "decode_cudagraph_manager") if getattr(r.speculator, k, None) is not None]:
        mg.graphs.clear()
    gc.collect(); r.capture_model()
verify = "verify" in open("/exp/gate.txt").read()
if mode == "report":
    RESULT = f"gate cnt(rows, set diff, diff gap>1e-2, sumerr_triton*1e3, sumerr_cublas*1e3, cublas set!=fp32, triton set!=fp32)={model._exp_gatecnt.tolist()}"
else:
    cnt = torch.zeros(7, dtype=torch.int64, device="cuda"); model._exp_gatecnt = cnt
    if mode == "on":
        for g in gates:
            orig = type(g).forward.__get__(g); w = g.weight
            def fwd(x, _w=w, _o=orig):
                if x.reshape(-1, x.shape[-1]).shape[0] > 16: return _o(x)
                y = BG.linear(x, _w, (8, 256, 1, 1))
                if verify:
                    ref = _o(x)[0].float(); yy = y.float()
                    sa = torch.sort(torch.topk(ref, 10, dim=-1).indices, -1).values
                    sb = torch.sort(torch.topk(yy, 10, dim=-1).indices, -1).values
                    diff = (sa != sb).any(-1)
                    t11 = torch.topk(ref, 11, dim=-1).values
                    gap = t11[:, 9] - t11[:, 10]
                    cnt[0] += ref.shape[0]; cnt[1] += diff.sum(); cnt[2] += (diff & (gap > 1e-2)).sum()
                    tru = x.reshape(-1, x.shape[-1]).float() @ _w.float().t()
                    ea = (ref - tru).abs(); eb = (yy - tru).abs()
                    cnt[3] += (eb.sum() * 1e3).long(); cnt[4] += (ea.sum() * 1e3).long()
                    st_ = torch.sort(torch.topk(tru, 10, dim=-1).indices, -1).values
                    cnt[5] += (sa != st_).any(-1).sum(); cnt[6] += (sb != st_).any(-1).sum()
                return y, None
            g.forward = fwd
    else:
        for g in gates: g.__dict__.pop("forward", None)
    recapture(); RESULT = f"gate {mode}"
