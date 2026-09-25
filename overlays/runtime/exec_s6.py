# S6 served toggle (/exp/s6.txt: on | off | stats). Sampled plain batches with T in [0.7,1.0], top_k==20, top_p==0.95 -> INT4-g32 top-1024
# shortlist + HX-kernel refine on gathered rows (bit-identical to E46 HX logits on the shortlist), -inf elsewhere. Everything else unchanged.
import builtins, importlib.util, torch, numpy as np
def ld(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
r = worker.model_runner; model = r.model; D = builtins.__dict__.setdefault("_exp_s6", {}); G = builtins._exp_hgate
mode = open("/exp/s6.txt").read().split()[0]; out = []
def uninstall():
    if "prev_cl" in D: model.compute_logits = D.pop("prev_cl"); r.sample = D.pop("prev_sample"); out.append("s6 removed")
if mode == "on":
    uninstall()
    h4 = ld("exp_head4", "/exp/exp_head4.py"); HX = ld("exp_hx", "/exp/exp_hx.py")
    lm = getattr(model, "lm_head", None) or model.language_model.lm_head; W = lm.weight; V = W.shape[0]
    if "i4" not in D: D["i4"] = h4.quantize_int4_g32(W)
    P, S = D["i4"]; SM, OF, GM, er, ec, ev = model._exp_hxP; C = 1024
    pos = torch.full((V,), -1, dtype=torch.long, device=W.device)
    FX = len(open("/exp/s6.txt").read().split()) > 1 and open("/exp/s6.txt").read().split()[1] == "fx"
    NOE = (er[:0], ec[:0], ev[:0])
    def fast_u(h):
        h = h.to(torch.bfloat16).contiguous(); m = h.shape[0]
        idx = h4.int4_logits(h, P, S, V)[:, :V].topk(C, 1).indices.reshape(-1).unique()
        pos.fill_(-1); pos[idx] = torch.arange(idx.numel(), device=W.device); pe = pos[er]; sel = pe >= 0
        y = HX.logits(h, (SM.index_select(0, idx), OF.index_select(0, idx), GM.index_select(0, idx), pe[sel], ec[sel], ev[sel])).to(torch.bfloat16)
        o = torch.full((m, V), float("-inf"), dtype=torch.bfloat16, device=h.device); o[:, idx] = y; return o
    def fast_fx(h):
        h = h.to(torch.bfloat16).contiguous(); m = h.shape[0]
        idx = h4.int4_logits(h, P, S, V)[:, :V].topk(C, 1).indices.reshape(-1)
        y = HX.logits(h, (SM.index_select(0, idx), OF.index_select(0, idx), GM.index_select(0, idx), *NOE))
        F = torch.zeros((m, V), dtype=torch.float32, device=h.device); F[:, idx] = y
        F.index_add_(1, er, h.float()[:, ec] * ev[None, :])
        mask = torch.zeros((V,), dtype=torch.bool, device=h.device).index_fill_(0, idx, True)
        return F.masked_fill_(~mask[None, :], float("-inf")).to(torch.bfloat16)
    fast = fast_fx if FX else fast_u
    ss = r.sampler.sampling_states; S_ = r.sampler; pen = S_.penalties_state; lb = S_.logit_bias_state; bw = S_.bad_words_state
    tb = S_.thinking_budget_state; lti = S_.logprob_token_ids_state
    st = D.setdefault("stats", {"s6": 0, "other": 0}); flag = {"ok": False}
    def ok(ib, go):
        if go is not None: return False
        ix = ib.idx_mapping_np
        if ix.size == 0: return False
        t = ss.temperature.np[ix]
        if not (np.all(t >= 0.7) and np.all(t <= 1.0)): return False
        if not np.all(ss.top_k.np[ix] == 20) or not np.all(np.abs(ss.top_p.np[ix] - 0.95) < 1e-6): return False
        if np.any(ss.min_p.np[ix] > 0): return False
        if np.any(pen.use_penalty[ix]) or np.any(lb.use_logit_bias[ix]): return False
        if bw.num_bad_words.np[ix].max(initial=0) > 0: return False
        if tb.enabled and np.any(tb.use_thinking_budget[ix]): return False
        if ss.num_logprobs[ix].max(initial=-1) != -1: return False
        if lti.num_token_ids.np[ix].max(initial=0) > 0: return False
        return True
    prev_sample = r.sample; prev_cl = model.compute_logits
    def sample(hs, ib, go):
        flag["ok"] = ok(ib, go)
        try: return prev_sample(hs, ib, go)
        finally: flag["ok"] = False
    def cl(hs, *a, **k):
        if flag["ok"] and 0 < hs.shape[0] <= 16: st["s6"] += 1; return fast(hs)
        st["other"] += 1; return prev_cl(hs, *a, **k)
    D["prev_cl"] = prev_cl; D["prev_sample"] = prev_sample; r.sample = sample; model.compute_logits = cl
    hh = torch.cat(builtins._exp_hsamp["rows"])[:4].contiguous(); a = fast(hh).float(); b = model._exp_orig_compute_logits(hh).float()[:, :V]
    fin = torch.isfinite(a); out.append(f"s6 on ({'fx' if FX else 'unique'}); self-check shortlist bit-equal {(a[fin] == b[fin]).all().item()}, argmax equal {(a.argmax(1) == b.argmax(1)).all().item()}")
elif mode == "off":
    uninstall(); D.pop("i4", None); torch.cuda.empty_cache()
out.append(f"stats {D.get('stats')}"); RESULT = "\n".join(out)
