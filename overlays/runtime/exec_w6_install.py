# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# Target dense projections: W6A16 g32 for decode (M<=16), original MXFP8 kept for prefill. Mode /exp/w6.txt: on | off
import importlib.util, torch, gc, time, builtins
s_ = importlib.util.spec_from_file_location("exp_w6", "/exp/exp_w6.py"); W6 = importlib.util.module_from_spec(s_); s_.loader.exec_module(W6)
s_ = importlib.util.spec_from_file_location("exp_mx8", "/exp/exp_mx8.py"); X8 = importlib.util.module_from_spec(s_); s_.loader.exec_module(X8)
r = worker.model_runner; model = r.model
mode = open("/exp/w6.txt").read().split()[0]
CFG = {"linear_attn.in_proj_qkvz": (16, 256, 1, 2), "linear_attn.out_proj": (16, 256, 1, 2),
       "self_attn.qkv_proj": (16, 256, 1, 4), "self_attn.o_proj": (16, 256, 1, 2)}
import os
if os.path.exists("/exp/w6_shared.txt") and open("/exp/w6_shared.txt").read().strip() == "1":
    CFG.update({"shared_expert.gate_up_proj": (16, 256, 1, 2), "shared_expert.down_proj": (16, 128, 1, 2)})
tg = [(n, m, s) for n, m in model.named_modules() for s in CFG if n.endswith(s) and "mtp" not in n]
def recapture():
    for mg in [r.cudagraph_manager] + [getattr(r.speculator, k) for k in ("prefill_cudagraph_manager", "decode_cudagraph_manager") if getattr(r.speculator, k, None) is not None]:
        mg.graphs.clear()
    gc.collect(); torch.cuda.empty_cache(); r.capture_model()
free0 = torch.cuda.mem_get_info()[0]
if mode == "on":
    nb = 0
    for n, m, s in tg:
        if getattr(m, "_exp_w6", None) is not None: continue
        N, K = m.weight.shape
        wf = (m.weight.data.float().view(N, K // 32, 32) * torch.exp2(X8.unswizzle(m.weight_scale.data, N, K).float() - 127)[..., None]).view(N, K)
        q = W6.quantize(wf); del wf
        m._exp_w6 = q; nb += sum(t.numel() * t.element_size() for t in q)
        orig = type(m).forward.__get__(m)
        bias = getattr(m, "bias", None); rb = getattr(m, "return_bias", True)
        def fwd(x, _q=q, _cfg=CFG[s], _orig=orig, _b=bias, _rb=rb):
            if x.reshape(-1, x.shape[-1]).shape[0] > builtins.__dict__.get("_exp_w6_maxm", 16):
                return _orig(x)
            y = W6.linear(x, *_q, _cfg)
            if _b is not None and not isinstance(_b, bool): y = y + _b
            return (y, None) if _rb else y
        m.forward = fwd
    t = time.time(); recapture()
    RESULT = f"W6 on: {len(tg)} modules, +{nb/1e9:.2f} GB, free {free0/2**30:.1f}->{torch.cuda.mem_get_info()[0]/2**30:.1f} GiB, recapture {time.time()-t:.1f}s"
elif mode == "allm":
    builtins._exp_w6_maxm = 1 << 30; RESULT = "W6 for all M (eval only)"
elif mode == "freemx":
    builtins._exp_w6_maxm = 1 << 30
    fb = 0
    for n, m, s in tg:
        if getattr(m, "_exp_w6", None) is None or m.weight.numel() == 0: continue
        fb += m.weight.numel() * m.weight.element_size() + m.weight_scale.numel()
        m.weight.data = torch.empty(0, dtype=m.weight.dtype, device=m.weight.device)
        m.weight_scale.data = torch.empty(0, dtype=m.weight_scale.dtype, device=m.weight_scale.device)
    recapture()
    RESULT = f"MXFP8 copies freed ({fb/1e9:.2f} GB); W6 serves all M; free {free0/2**30:.1f}->{torch.cuda.mem_get_info()[0]/2**30:.1f} GiB (irreversible until restart)"
elif mode == "decode":
    builtins._exp_w6_maxm = 16; RESULT = "W6 decode only"
else:
    for n, m, s in tg:
        m.__dict__.pop("forward", None); m._exp_w6 = None
    recapture(); RESULT = "W6 off"
