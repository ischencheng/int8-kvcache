"""Host overhead check: the same decode call timed eagerly and replayed from a CUDA graph."""
import json

import torch
from bench_kernels import SCALE, decode_case, segm, static_for

from kvq import cuda_kernels as ck
from kvq import quant
from kvq.triton_kernels.int8_attention import paged_attention


def timed(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


def graphed(fn):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


for batch, ctx in [(1, 4096), (32, 4096), (64, 4096)]:
    q, k, v, cu, sl, bt = decode_case(batch, ctx)
    st = static_for(k, v)
    kq = quant.quantize(k.float() - st["k_mu"], st["k_scale"])
    vq = quant.quantize(v, st["v_scale"])
    out = torch.empty_like(q)
    sg = segm(batch)
    fns = {"triton_int8": lambda: paged_attention(q, kq, vq, out, cu, sl, bt, 1, SCALE, st, segm=sg),
           "cuda_int8": lambda: ck.decode_attention(q, kq, vq, out, cu, sl, bt, SCALE, st, segm=sg),
           "cuda_int8_nosplit": lambda: ck.decode_attention(q, kq, vq, out, cu, sl, bt, SCALE, st)}
    row = {"batch": batch, "ctx": ctx}
    for name, fn in fns.items():
        row[f"{name}_eager"] = round(timed(fn), 1)
        row[f"{name}_graph"] = round(timed(graphed(fn)), 1)
    print(json.dumps(row), flush=True)
