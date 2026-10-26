"""A/B of the INT8 attention for decode (q_len 1) and verification batches (q_len 5) on one GPU:
the previous kernel (built from scripts/_prev_kernel/int8_kv.cu when present, e.g.
`git show 52d0931:kvq/cuda/int8_kv.cu`), the current one, and FA2 on a bf16 cache."""
import json
import os

import torch
from bench_kernels import HQ, SCALE, THRESH_3D, D, bench, static_for
from bench_short_q import case
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

from kvq import cuda_kernels as ck
from kvq import quant

PREV = os.path.join(os.path.dirname(__file__), "_prev_kernel", "int8_kv.cu")
prev = None
if os.path.exists(PREV):
    from torch.utils.cpp_extension import load

    ck.ext()  # sets TORCH_CUDA_ARCH_LIST for this GPU
    prev = load(name="kvq_int8_kv_prev", sources=[PREV], extra_cuda_cflags=["-O3"], verbose=False)

for batch, ctx in [(1, 512), (1, 1024), (1, 4096), (8, 1024), (32, 1024), (32, 4096)]:
    for q_len in (1, 5):
        q, k, v, cu, sl, bt = case(batch, ctx, q_len)
        st = static_for(k, v)
        kq = quant.quantize(k.float() - st["k_mu"], st["k_scale"])
        vq = quant.quantize(v, st["v_scale"])
        out = torch.empty_like(q)
        segm = None
        if batch <= THRESH_3D and q.shape[0] <= THRESH_3D:  # the backend's split-KV rule
            segm = (torch.empty(THRESH_3D, HQ, 16, D, device="cuda"), torch.empty(THRESH_3D, HQ, 16, device="cuda"),
                    torch.empty(THRESH_3D, HQ, 16, device="cuda"))
        so, sm, ss = segm if segm else (None, None, None)
        row = {"batch": batch, "ctx": ctx, "q_len": q_len, "split": segm is not None}
        row["fa2"] = bench(lambda: flash_attn_varlen_func(
            q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=q_len, seqused_k=sl, max_seqlen_k=ctx,
            softmax_scale=SCALE, causal=True, block_table=bt, fa_version=2))
        if prev is not None:
            row["prev"] = bench(lambda: prev.int8_decode_attention(q, kq, vq, out, cu, sl, bt, st["k_scale"],
                                                                   st["v_scale"], SCALE, so, sm, ss, q_len, 8))
        row["int8"] = bench(lambda: ck.decode_attention(q, kq, vq, out, cu, sl, bt, SCALE, st, segm=segm,
                                                        max_query_len=q_len))
        print(json.dumps({k_: round(v_, 1) if isinstance(v_, float) else v_ for k_, v_ in row.items()}), flush=True)
