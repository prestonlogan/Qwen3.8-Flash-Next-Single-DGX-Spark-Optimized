# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# PLE zero-copy handshake (requires container launched with ple_worker_zc.py mounted as the PLE worker).
# /exp/ple_zc.txt: timing | on | off | report
#  timing: worker keeps the DMA path; records per-step cross-process timestamps into the shared header.
#  on:     worker writes rows straight into host-shared /dev/shm/exp_ple_zc; graph spins on the done flag
#          and then copies rows host-mapped -> GPU buffer inside the same kernel (no CPU-side CUDA work).
import torch, triton, triton.language as tl, builtins, gc, os, time, statistics as st
import numpy as np
from vllm.v1.ple_offload.protocol import PleOffloadRequest
r = worker.model_runner; c = r._ple_offload_connector
mode = open("/exp/ple_zc.txt").read().split()[0]
PA = builtins._exp_pa            # from exec_ple_async.py (flag registration, expv)
S = builtins.__dict__.setdefault("_exp_zc", {})
PATH, HDR = "/dev/shm/exp_ple_zc", 4096
layer = next(iter(c._layers.values())); gbuf = layer._gpu_output_buffer
nbytes = gbuf.numel() * gbuf.element_size()
if "mm" not in S:
    if not os.path.exists(PATH) or os.path.getsize(PATH) < HDR + nbytes:
        with open(PATH, "wb") as f: f.truncate(HDR + nbytes)
    mm = np.memmap(PATH, dtype=np.uint8, mode="r+")
    S["mm"] = mm; S["hdr"] = torch.from_numpy(mm[:HDR].view(np.int64))
    S["host"] = torch.from_numpy(mm[HDR:HDR + nbytes])
    rc = torch.cuda.cudart().cudaHostRegister(S["host"].data_ptr(), nbytes, 2)
    S["reg"] = str(rc); S["rows"] = []
hdr = S["hdr"]

@triton.jit
def _spin_copy(FLAG, EXP, SRC, DST, W: tl.constexpr, BW: tl.constexpr, LIMIT: tl.constexpr):
    row = tl.program_id(0)
    e = tl.load(EXP)
    v = tl.load(FLAG, volatile=True)
    n = 0
    while (v < e) & (n < LIMIT):
        v = tl.load(FLAG, volatile=True)
        n += 1
    v = tl.atomic_add(FLAG, 0, sem="acquire", scope="sys")   # acquire: row reads below cannot pass the flag read
    if (e > 0) & (v >= e):
        o = tl.arange(0, BW)
        x = tl.load(SRC + row * W + o, o < W, volatile=True)
        tl.store(DST + row * W + o, x, o < W)

class _Ptr:
    def __init__(self, p, dt): self.p = p; self.dtype = dt
    def data_ptr(self): return self.p

def recapture():
    for mg in [r.cudagraph_manager, r.speculator.prefill_cudagraph_manager, r.speculator.decode_cudagraph_manager]: mg.graphs.clear()
    gc.collect(); r.capture_model()

expv, fp = PA["exp"], PA["fp"]
W8 = gbuf.shape[1] * gbuf.element_size() // 8
src = _Ptr(S["host"].data_ptr(), torch.int64)
dst64 = gbuf.view(torch.uint8).view(-1).view(torch.int64).view(gbuf.shape[0], W8)
if "launch0" not in S: S["launch0"] = c._launch      # the async launch from exec_ple_async
base = S["launch0"]
if mode != "report": S["chk"] = [0, 0, 0]; S["bad"] = []
import ctypes
_dmb = ctypes.CDLL("/exp/exp_dmb.so").exp_dmb
def launch_shm(num_reqs, num_tokens):
    # replaces the async launch: stage inputs, publish the request in the shm slot (seq last), then zmq as fallback
    c._seq += 1; seq = c._seq
    c._input_ready_event.record(torch.cuda.current_stream(c.device))
    req = PleOffloadRequest(dp_rank=c.dp_rank, num_tokens=num_tokens, num_reqs=num_reqs, seq=seq)
    c._copy_cuda_inputs(req)
    hdr[16] = c.dp_rank; hdr[17] = num_tokens; hdr[18] = num_reqs
    _dmb(); hdr[20] = seq
    import msgspec
    c._request_socket.send(msgspec.msgpack.encode(req))
    expv.fill_(seq)
def launch_t(num_reqs, num_tokens):
    if S.get("verify") and S.get("last"):
        n = S["last"][2]
        if n <= 16 and S.get("lastexp") == S["last"][1]:
            torch.cuda.synchronize()
            g = dst64[:n].cpu(); h = S["host"].view(torch.int64).view(-1, W8)[:n]
            S["chk"][0] += 1; bad = not torch.equal(g, h); S["chk"][1] += int(bad)
            if bad: S.setdefault("bad", []).append((n, int(expv.item()), int(S['last'][1]), int((g != h).any(1).sum()), int(hdr[11])))
            S["chk"][2] += int(h.abs().sum().item() == 0)
    t = time.monotonic_ns()
    if S.get("last") and int(hdr[11]) == S["last"][1]:
        t0 = S["last"][0]
        S["rows"].append(((int(hdr[8]) - t0) / 1e6, (int(hdr[9]) - int(hdr[8])) / 1e6, (int(hdr[10]) - int(hdr[9])) / 1e6, S["last"][2]))
    (launch_shm if S.get("shm") else base)(num_reqs, num_tokens)
    S["last"] = (t, c._seq, num_tokens)
    S["lastexp"] = c._seq if S.get("verify") else None
if mode != "report": S["verify"] = "verify" in open("/exp/ple_zc.txt").read()
if "sdo0" not in S: S["sdo0"] = c.signal_dummy_outputs
def _sdo(num_tokens):
    S["lastexp"] = None; S["sdo0"](num_tokens)
c.signal_dummy_outputs = _sdo
if mode != "report":
    S["shm"] = "shm" in open("/exp/ple_zc.txt").read()
    hdr[3] = 1 if S["shm"] else 0
if mode in ("timing", "on"):
    hdr[0] = 1 if mode == "on" else 0
    c._launch = launch_t
    if mode == "on":
        for name, l in c._layers.items():
            def fwd(hidden_states, input_ids, *a, _l=l, **k):
                n = input_ids.shape[0]
                _spin_copy[(n,)](fp, expv, src, dst64, W8, triton.next_power_of_2(W8), 50_000_000)
                return _l._gpu_output_buffer[:n]
            l.forward = fwd
        recapture()
    RESULT = f"{mode}: host reg {S['reg']}, W8={W8}"
elif mode == "report":
    d = [x for x in S["rows"] if x[3] <= 16]
    def q(i): v = sorted(x[i] for x in d); return f"med {st.median(v):.3f} p90 {v[int(len(v)*.9)]:.3f}"
    RESULT = f"shm_handled={int(hdr[21])} zmq_handled={int(hdr[22])}; chk(steps, mismatched, all-zero)={S['chk']} bad={S.get('bad', [])[-5:]}; n={len(d)} launch->worker_recv {q(0)} | recv->dma_done {q(1)} | fence+flag {q(2)} ms"
    S["rows"] = []
else:
    hdr[0] = 0; hdr[3] = 0; S["shm"] = False; c._launch = base
    for name, l in c._layers.items():
        l.__dict__.pop("forward", None)
    RESULT = "zc off: worker on DMA path; NOW run exec_ple_async.py (ple_async.txt=on) to restore the async forward + recapture"
