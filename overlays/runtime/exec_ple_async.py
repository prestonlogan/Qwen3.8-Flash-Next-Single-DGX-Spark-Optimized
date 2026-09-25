# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# Async PLE handshake: host no longer blocks in _wait_done; an in-graph Triton spin on the host-mapped done flag
# (cudaHostRegister) gates the PLE placeholder. GPU runs everything before the PLE layer while the CPU worker gathers.
# Mode /exp/ple_async.txt: on | off
import torch, triton, triton.language as tl, builtins, gc, time
from vllm.v1.ple_offload.protocol import PleOffloadRequest
r = worker.model_runner; c = r._ple_offload_connector
mode = open("/exp/ple_async.txt").read().split()[0]

@triton.jit
def _ple_spin(FLAG, EXP, LIMIT: tl.constexpr):
    e = tl.load(EXP)
    v = tl.load(FLAG, volatile=True)
    n = 0
    while (v < e) & (n < LIMIT):
        v = tl.load(FLAG, volatile=True)
        n += 1

class _Ptr:
    dtype = torch.int64
    def __init__(self, p): self.p = p
    def data_ptr(self): return self.p

def recapture():
    for mg in [r.cudagraph_manager] + [getattr(r.speculator, k) for k in ("prefill_cudagraph_manager", "decode_cudagraph_manager") if getattr(r.speculator, k, None) is not None]:
        mg.graphs.clear()
    gc.collect(); r.capture_model()

st = builtins.__dict__.setdefault("_exp_pa", {})
if mode == "on":
    if "reg" not in st:
        rc = torch.cuda.cudart().cudaHostRegister(c._done_flag.data_ptr(), 8, 2)
        st["reg"] = str(rc)
        st["exp"] = torch.zeros(1, dtype=torch.int64, device="cuda")
        st["fp"] = _Ptr(c._done_flag.data_ptr())
    expv, fp = st["exp"], st["fp"]
    expv.zero_()
    for name, layer in c._layers.items():
        def fwd(hidden_states, input_ids, *a, _l=layer, **k):
            _ple_spin[(1,)](fp, expv, 50_000_000)
            return _l._gpu_output_buffer[: input_ids.shape[0]]
        layer.forward = fwd
    def launch(num_reqs, num_tokens):
        c._seq += 1; seq = c._seq
        c._input_ready_event.record(torch.cuda.current_stream(c.device))
        c._process_request(PleOffloadRequest(dp_rank=c.dp_rank, num_tokens=num_tokens, num_reqs=num_reqs, seq=seq), c._request_socket)
        expv.fill_(seq)
    c._launch = launch
    c.__dict__.pop("signal_dummy_outputs", None); osd = c.signal_dummy_outputs
    def sdo(num_tokens):
        expv.zero_(); osd(num_tokens)
    c.signal_dummy_outputs = sdo
    t = time.time(); recapture()
    RESULT = f"ple async ON (hostRegister {st['reg']}), {len(c._layers)} PLE layer(s), recapture {time.time()-t:.1f}s"
else:
    for name, layer in c._layers.items():
        layer.__dict__.pop("forward", None)
    c.__dict__.pop("_launch", None); c.__dict__.pop("signal_dummy_outputs", None)
    recapture()
    RESULT = "ple async OFF"
