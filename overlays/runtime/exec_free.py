# Free unused coarse copies: target INT4 (_exp_int4) and draft INT4 (_exp_draft_int4) if the INT2 paths are live.
import torch, gc, sys
r = worker.model_runner; model = r.model; m = r.speculator.model
out = []
f0 = torch.cuda.mem_get_info()[0]
if hasattr(model, "_exp_int2") and hasattr(model, "_exp_int4"):
    del model._exp_int4; out.append("target int4 freed")
if hasattr(m, "_exp_draft_int2") and hasattr(m, "_exp_draft_int4"):
    mod = sys.modules[type(m).__module__]
    if mod._exp_fp8_logits is not getattr(mod, "_exp_int4r_fn", None):
        mod._exp_int4r_fn = None; del m._exp_draft_int4; out.append("draft int4 freed")
gc.collect(); torch.cuda.empty_cache()
out.append(f"free {f0/2**30:.2f} -> {torch.cuda.mem_get_info()[0]/2**30:.2f} GiB")
RESULT = "; ".join(out)
