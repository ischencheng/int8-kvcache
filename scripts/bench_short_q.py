"""Kernel time of a spec-decode verification batch (q_len = k + 1 = 5) vs decode (q_len 1):
our INT8 kernel against FA2 on a bf16 cache, Qwen2.5-7B shapes."""
import json

import torch
from bench_kernels import BS, D, HKV, HQ, SCALE, THRESH_3D, bench, static_for
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

from kvq import cuda_kernels as ck
from kvq import quant


def case(batch, ctx, q_len, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = (ctx + BS - 1) // BS
    nb = batch * pages + 1
    bt = (torch.randperm(nb - 1, generator=g, device="cuda")[: batch * pages] + 1).to(torch.int32).view(batch, pages)
    q = torch.randn(batch * q_len, HQ, D, generator=g, device="cuda").to(torch.bfloat16)
    k = torch.randn(nb, BS, HKV, D, generator=g, device="cuda").to(torch.bfloat16)
    v = torch.randn(nb, BS, HKV, D, generator=g, device="cuda").to(torch.bfloat16)
    cu = torch.arange(0, batch * q_len + 1, q_len, dtype=torch.int32, device="cuda")
    sl = torch.full((batch,), ctx, dtype=torch.int32, device="cuda")
    return q, k, v, cu, sl, bt


if __name__ == "__main__":
    for batch, ctx in [(1, 512), (1, 1024), (1, 4096), (8, 1024), (32, 1024)]:
        row = {"batch": batch, "ctx": ctx}
        for q_len in (1, 5):
            q, k, v, cu, sl, bt = case(batch, ctx, q_len)
            st = static_for(k, v)
            kq = quant.quantize(k.float() - st["k_mu"], st["k_scale"])
            vq = quant.quantize(v, st["v_scale"])
            out = torch.empty_like(q)
            n = q.shape[0]
            # the backend's split-KV rule: few requests and the token rows fit vLLM's scratch
            segm = None
            if batch <= THRESH_3D and n <= THRESH_3D:
                segm = (torch.empty(THRESH_3D, HQ, 16, D, device="cuda"), torch.empty(THRESH_3D, HQ, 16, device="cuda"),
                        torch.empty(THRESH_3D, HQ, 16, device="cuda"))
            row[f"fa2_q{q_len}"] = bench(lambda: flash_attn_varlen_func(
                q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=q_len, seqused_k=sl, max_seqlen_k=ctx,
                softmax_scale=SCALE, causal=True, block_table=bt, fa_version=2))
            row[f"int8_q{q_len}"] = bench(lambda: ck.decode_attention(q, kq, vq, out, cu, sl, bt, SCALE, st, segm=segm,
                                                                      max_query_len=q_len))
        print(json.dumps({k_: round(v_, 1) if isinstance(v_, float) else v_ for k_, v_ in row.items()}), flush=True)
