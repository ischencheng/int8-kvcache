"""Python side of the CUDA kernels in kvq/cuda/int8_kv.cu (JIT-compiled on first use).

Attention on the INT8 cache is split by query length:
  * requests with at most SHORT_Q query tokens (decode, spec-decode verification) run the INT8
    tensor-core kernel (implementation B);
  * longer ones (prefill, chunked prefill) have their pages dequantized into a bf16 paged cache
    that vLLM's FlashAttention reads (implementation A). The page table of that temporary cache
    comes from plan_long(), once per step for all layers.
In a mixed batch FlashAttention runs first over every request with the short ones given zero
keys, then the INT8 kernel overwrites the short requests' rows: no reordering of the batch.
"""
import os
from dataclasses import dataclass

import numpy as np
import torch

SHORT_Q = 8  # q_len at or below this: INT8 kernel; above: dequantize + FlashAttention

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        major, minor = torch.cuda.get_device_capability()
        # The vLLM image presets 7 architectures for its own builds; compile only for this GPU.
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        src = os.path.join(os.path.dirname(__file__), "cuda", "int8_kv.cu")
        _EXT = load(name="kvq_int8_kv", sources=[src], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
    return _EXT


def quant_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, static):
    ext().quant_cache(key, value, key_cache, value_cache, slot_mapping, static["k_scale"], static["v_scale"],
                      static["k_mu"])


def quant_q(q, k_scale, group=None):
    return tuple(ext().quant_q(q, k_scale, group or q.shape[-1]))


def decode_attention(q, k_cache, v_cache, out, cu_seqlens_q, seq_lens, block_table, softmax_scale, static,
                     segm=None, max_query_len=1):
    """INT8 attention for the requests with at most SHORT_Q query tokens (others are skipped).
    max_query_len bounds the short queries' length (it sets the number of m tiles)."""
    so, sm, ss = segm if segm is not None else (None, None, None)
    ext().int8_decode_attention(q, k_cache, v_cache, out, cu_seqlens_q, seq_lens, block_table, static["k_scale"],
                                static["v_scale"], softmax_scale, so, sm, ss, max_query_len, SHORT_Q)
    return out


@dataclass
class LongPlan:
    """Temporary bf16 paged cache for the long-query requests of one step."""
    src_pages: torch.Tensor     # [P] int32: INT8 cache page copied into temporary page p
    block_table: torch.Tensor   # [num_reqs, max_pages] int32 into the temporary cache (rows of short requests unused)
    seqused_k: torch.Tensor     # [num_reqs] int32: key count for FlashAttention, 0 for short requests
    max_seqlen_k: int
    has_short: bool             # any request left for the INT8 kernel
    block_size: int             # page size of both caches


def plan_long(query_start_loc_cpu, seq_lens_cpu, block_table, block_size):
    """Page table of the temporary cache, or None when no request has more than SHORT_Q query tokens.
    Host-side lengths (vLLM's CPU copies) size the cache, so this runs only outside full CUDA graphs."""
    qsl = np.asarray(query_start_loc_cpu, dtype=np.int64)
    num_reqs = len(qsl) - 1
    q_lens = np.diff(qsl)
    long = q_lens > SHORT_Q
    if not long.any():
        return None
    seq = np.asarray(seq_lens_cpu, dtype=np.int64)[:num_reqs]
    pages = np.where(long, -(-seq // block_size), 0)
    off = np.cumsum(pages) - pages
    total = int(pages.sum())
    src_req = np.repeat(np.arange(num_reqs), pages)
    src_j = np.arange(total) - np.repeat(off, pages)
    bt = np.zeros((num_reqs, int(pages.max())), dtype=np.int32)
    bt[src_req, src_j] = np.arange(total)
    dev = block_table.device
    src_pages = block_table[torch.from_numpy(src_req).to(dev), torch.from_numpy(src_j).to(dev)].to(torch.int32)
    return LongPlan(src_pages=src_pages, block_table=torch.from_numpy(bt).to(dev),
                    seqused_k=torch.from_numpy(np.where(long, seq, 0).astype(np.int32)).to(dev),
                    max_seqlen_k=int(seq[long].max()), has_short=bool((q_lens[~long] > 0).any()),
                    block_size=block_size)


def long_attention(q, k_cache, v_cache, out, cu_seqlens_q, max_query_len, plan, softmax_scale, static):
    """Dequantize the long requests' pages, then FlashAttention over the whole batch (short requests
    have zero keys there and are overwritten by decode_attention afterwards)."""
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, get_flash_attn_version

    bs, h, d = k_cache.shape[1:]
    assert bs == plan.block_size, (bs, plan.block_size)
    kt = torch.empty(plan.src_pages.shape[0], bs, h, d, dtype=q.dtype, device=q.device)
    vt = torch.empty_like(kt)
    ext().dequant_pages(k_cache, v_cache, kt, vt, plan.src_pages, static["k_scale"], static["v_scale"])
    flash_attn_varlen_func(q=q, k=kt, v=vt, out=out, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_query_len,
                           seqused_k=plan.seqused_k, max_seqlen_k=plan.max_seqlen_k, softmax_scale=softmax_scale,
                           causal=True, block_table=plan.block_table, fa_version=get_flash_attn_version())
    return out


def paged_attention(q, k_cache, v_cache, out, cu_seqlens_q, seq_lens, block_table, max_query_len, softmax_scale,
                    static, plan=None, segm=None):
    """Both paths for one batch, as the backend runs them (plan from plan_long, None if all short)."""
    if plan is not None:
        long_attention(q, k_cache, v_cache, out, cu_seqlens_q, max_query_len, plan, softmax_scale, static)
    if plan is None or plan.has_short:
        decode_attention(q, k_cache, v_cache, out, cu_seqlens_q, seq_lens, block_table, softmax_scale, static,
                         segm, max_query_len)
    return out
