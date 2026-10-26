"""Per-kernel GPU time of the INT8 attention for decode (q_len 1) and verification batches (q_len 5):
where the extra time of q_len 5 goes (attention kernel vs split-KV reduce), with FA2 for reference."""
import json
import re
from collections import defaultdict

import torch
from bench_kernels import HQ, SCALE, THRESH_3D, D, static_for
from bench_short_q import case
from torch.profiler import ProfilerActivity, profile
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

from kvq import cuda_kernels as ck
from kvq import quant

ITERS = 50


def kernel_times(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
    t = defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            m = re.search(r"(\w+_kernel)", e.name)
            name = m.group(1) if m else e.name[:40]
            t[name] += e.device_time / ITERS
    return {k: round(v, 1) for k, v in t.items()}


for batch, ctx in [(1, 512), (1, 1024), (8, 1024)]:
    for q_len in (1, 5):
        q, k, v, cu, sl, bt = case(batch, ctx, q_len)
        st = static_for(k, v)
        kq = quant.quantize(k.float() - st["k_mu"], st["k_scale"])
        vq = quant.quantize(v, st["v_scale"])
        out = torch.empty_like(q)
        segm = None
        if batch <= THRESH_3D and q.shape[0] <= THRESH_3D:
            segm = (torch.empty(THRESH_3D, HQ, 16, D, device="cuda"), torch.empty(THRESH_3D, HQ, 16, device="cuda"),
                    torch.empty(THRESH_3D, HQ, 16, device="cuda"))
        row = {"batch": batch, "ctx": ctx, "q_len": q_len,
               "int8": kernel_times(lambda: ck.decode_attention(q, kq, vq, out, cu, sl, bt, SCALE, st, segm=segm,
                                                                max_query_len=q_len)),
               "fa2": kernel_times(lambda: flash_attn_varlen_func(
                   q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=q_len, seqused_k=sl, max_seqlen_k=ctx,
                   softmax_scale=SCALE, causal=True, block_table=bt, fa_version=2))}
        print(json.dumps(row), flush=True)
