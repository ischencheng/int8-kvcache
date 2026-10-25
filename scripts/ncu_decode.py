"""One decode attention call per implementation, for Nsight Compute (scripts/ncu_report.sh).

Qwen2.5-7B shapes, batch 64 x 4096 context (single pass, no split-KV): the bandwidth-bound regime.
"""
import torch
from bench_kernels import SCALE, decode_case, static_for
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

from kvq import cuda_kernels as ck
from kvq import quant
from kvq.triton_kernels.int8_attention import paged_attention

batch, ctx = 64, 4096
q, k, v, cu, sl, bt = decode_case(batch, ctx)
st = static_for(k, v)
kq = quant.quantize(k.float() - st["k_mu"], st["k_scale"])
vq = quant.quantize(v, st["v_scale"])
out = torch.empty_like(q)
fns = [
    lambda: flash_attn_varlen_func(q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=sl,
                                   max_seqlen_k=ctx, softmax_scale=SCALE, causal=True, block_table=bt, fa_version=2),
    lambda: paged_attention(q, k, v, out, cu, sl, bt, 1, SCALE),
    lambda: paged_attention(q, kq, vq, out, cu, sl, bt, 1, SCALE, st),
    lambda: ck.decode_attention(q, kq, vq, out, cu, sl, bt, SCALE, st),
]
for fn in fns:   # compile / warm up outside the profiled range
    fn()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for fn in fns:
    fn()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("kv bytes bf16", batch * ctx * 4 * 128 * 2 * 2, "int8", batch * ctx * 4 * 128 * 2)
