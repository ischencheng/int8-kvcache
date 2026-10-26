"""Quick A/B of the decode kernels on a few batch x context points (kernel iteration loop)."""
import json
import sys

import torch
from bench_kernels import SCALE, bench, decode_case, segm, static_for
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

from kvq import cuda_kernels as ck
from kvq import quant

points = [(1, 4096), (8, 16384), (32, 4096), (32, 32768), (64, 4096), (64, 16384), (128, 4096), (256, 1024)]
for batch, ctx in points:
    q, k, v, cu, sl, bt = decode_case(batch, ctx)
    st = static_for(k, v)
    kq = quant.quantize(k.float() - st["k_mu"], st["k_scale"])
    vq = quant.quantize(v, st["v_scale"])
    out = torch.empty_like(q)
    sg = segm(batch)
    row = {"batch": batch, "ctx": ctx,
           "fa2_bf16": bench(lambda: flash_attn_varlen_func(q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=1,
                                                            seqused_k=sl, max_seqlen_k=ctx, softmax_scale=SCALE,
                                                            causal=True, block_table=bt, fa_version=2)),
           "cuda_int8": bench(lambda: ck.decode_attention(q, kq, vq, out, cu, sl, bt, SCALE, st, segm=sg))}
    row["GBps_cuda"] = batch * ctx * 4 * 128 * 2 / (row["cuda_int8"] * 1e-6) / 1e9
    print(json.dumps({k_: round(v_, 1) if isinstance(v_, float) else v_ for k_, v_ in row.items()}), flush=True)
