# QF1 install: route target-model routed experts (48 layers) to the SSHdotCodes NVFP4 decode MoE for 1..8 tokens.
# /exp/qfmoe.txt on|off. Falls back to the original forward_modular outside the gate. Recaptures graphs.
import torch, importlib.util, builtins, gc
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder
r = worker.model_runner; lm = r.model.language_model.model
mode = open("/exp/qfmoe.txt").read().split()[0]
if not hasattr(builtins, "_qf_moe"):
    sp = importlib.util.spec_from_file_location("qf_moe", "/exp/qf/build/qf_moe.so"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); builtins._qf_moe = m
qf = builtins._qf_moe
n_on = 0; notes = []
for li, l in enumerate(lm.layers):
    e = l.mlp.experts.routed_experts
    e.__dict__.pop("forward_modular", None)
    if mode != "on": continue
    c = e.quant_method.moe_quant_config
    a1 = c._a1.alpha_or_gscale; a2 = c._a2.alpha_or_gscale
    assert bool((a1 == a1[0]).all()) and bool((a2 == a2[0]).all()), f"layer {li}: per-expert input scales differ"
    assert c.is_scale_swizzled and e.w13_weight.shape == (512, 1280, 1280) and e.w2_weight.shape == (512, 2560, 320)
    A = (e.w13_weight, e.w13_weight_scale, c._w1.alpha_or_gscale.contiguous(), a1[:1].contiguous(),
         e.w2_weight, e.w2_weight_scale, c._w2.alpha_or_gscale.contiguous(), a2[:1].contiguous())
    orig = type(e).forward_modular.__get__(e)
    def fm(x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None, _A=A, _o=orig):
        if x.dim() == 2 and 1 <= x.shape[0] <= 8 and x.shape[1] == 2560 and x.dtype == torch.bfloat16 and topk_ids.shape[-1] == 10:
            if shared_experts is not None:
                shared_experts(shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
            ti = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
            tw = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.float()
            return qf.nvfp4_moe_decode(x.contiguous(), ti.contiguous(), tw.contiguous(), *_A)
        return _o(x=x, topk_weights=topk_weights, topk_ids=topk_ids, shared_experts=shared_experts, shared_experts_input=shared_experts_input)
    e.forward_modular = fm; n_on += 1
for mg in [r.cudagraph_manager, r.speculator.prefill_cudagraph_manager, r.speculator.decode_cudagraph_manager]: mg.graphs.clear()
gc.collect(); r.capture_model()
RESULT = f"qfmoe {mode}: layers={n_on}"
