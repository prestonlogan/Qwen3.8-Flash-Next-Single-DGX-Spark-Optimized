# HX2 served install: route the EXACT (sampled / non-fast) head through the lossless exponent-coded kernel.
# /exp/hx.txt: on | off | check.  Re-installs the E39 gate (head_gate on 512) so its closure picks up the new exact head.
import torch, builtins, importlib.util, gc
s_ = importlib.util.spec_from_file_location("exp_hx", "/exp/exp_hx.py"); HX = importlib.util.module_from_spec(s_); s_.loader.exec_module(HX)
r = worker.model_runner; model = r.model
lm = getattr(model, "lm_head", None) or model.language_model.lm_head; W = lm.weight
mode = open("/exp/hx.txt").read().split()[0]; out = []
if not hasattr(model, "_exp_hx_true_orig"): model._exp_hx_true_orig = model._exp_orig_compute_logits
true_orig = model._exp_hx_true_orig
def regate():
    open_ = open("/exp/head_gate.txt").read()
    exec(compile(open("/exp/exec_head_gate.py").read(), "gate", "exec"), {"worker": worker, "__name__": "g"})
if mode in ("on", "check"):
    if not hasattr(model, "_exp_hxP"):
        f0 = torch.cuda.mem_get_info()[0]; model._exp_hxP = HX.compress(W); torch.cuda.synchronize()
        out.append(f"compressed: escapes {model._exp_hxP[3].numel()}; alloc {(f0 - torch.cuda.mem_get_info()[0]) / 1e9:.3f} GB")
    P = model._exp_hxP
    def hx_logits(hidden_states, *a, **k):
        m = hidden_states.shape[0]
        if m == 0 or m > 16: return true_orig(hidden_states, *a, **k)
        ref_shape_V = None
        y = HX.logits(hidden_states.to(torch.bfloat16).contiguous(), P).to(torch.bfloat16)
        return y
    # correctness vs the true exact head on random + real-scale inputs
    x = torch.randn(4, W.shape[1], dtype=torch.bfloat16, device="cuda") * 2
    a = true_orig(x); b = hx_logits(x)
    out.append(f"shape exact {tuple(a.shape)} {a.dtype} hx {tuple(b.shape)} {b.dtype}")
    V = min(a.shape[1], b.shape[1]); af, bf = a[:, :V].float(), b[:, :V].float()
    ulp = ((af - bf).abs() / af.abs().clamp(min=1e-3)).max().item()
    fp32 = (x.float() @ W.float().T)
    out.append(f"argmax equal {bool((af.argmax(1) == bf.argmax(1)).all())}; max rel diff {ulp:.2e}; bitwise-equal frac {(a[:, :V] == b[:, :V]).float().mean().item():.4f}; "
               f"err vs fp32 exact: cublas {(af - fp32).abs().max().item():.3e} hx {(bf - fp32).abs().max().item():.3e}")
    for T in (1.0, 0.7):
        pa = (af / T).softmax(1); pb = (bf / T).softmax(1); out.append(f"TV T{T} max {0.5 * (pa - pb).abs().sum(1).max().item():.2e}")
    if mode == "on":
        model._exp_orig_compute_logits = hx_logits; regate(); out.append("exact head -> HX; gate reinstalled")
elif mode == "off":
    model._exp_orig_compute_logits = true_orig; regate(); out.append("exact head -> BF16 cuBLAS; gate reinstalled")
st = builtins.__dict__.get("_exp_hgate", {}).get("stats"); out.append(f"gate stats {st}")
RESULT = "\n".join(out)
