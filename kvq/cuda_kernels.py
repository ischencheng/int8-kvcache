"""Python side of the CUDA kernels in kvq/cuda/int8_kv.cu (JIT-compiled on first use).

Attention on the INT8 cache is split by query length:
  * requests with at most SHORT_Q query tokens (decode, spec-decode verification) run the INT8
    tensor-core kernel (implementation B);
  * longer ones (prefill, chunked prefill) have their pages dequantized into a bf16 paged cache
    that vLLM's FlashAttention reads (implementation A). The page table of that temporary cache
    comes from plan_long(), once per step for all layers.
In a mixed batch FlashAttention runs first over every request, the short ones reading as many keys
as they have query tokens from temporary page 0, then the INT8 kernel overwrites the short requests'
rows: no reordering of the batch.
"""
import os
from dataclasses import dataclass

import numpy as np
import torch

# q_len at or below this: INT8 kernel; above: dequantize + FlashAttention. KVQ_SHORT_Q=1000000 puts
# every row on the INT8 kernel (teacher-forced NLL of the decode numerics; slow for long prompts).
SHORT_Q = int(os.environ.get("KVQ_SHORT_Q", 8))

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
    """Temporary bf16 paged cache for the long-query requests of one step. Built on the host in the
    metadata builder; the GPU copies are made by the first layer's forward (on the forward stream),
    and the INT8 cache page ids are read from the step's block table by the dequantize kernel."""
    src_req: np.ndarray         # [P] request of temporary page p
    src_j: np.ndarray           # [P] page index of temporary page p inside that request
    block_table: np.ndarray     # [num_reqs, max_pages] into the temporary cache (rows of short requests unused)
    seqused_k: np.ndarray       # [num_reqs] key count for FlashAttention (short requests: their q_len)
    max_seqlen_k: int
    has_short: bool             # any request left for the INT8 kernel
    block_size: int             # page size of both caches
    _gpu: tuple = None

    def gpu(self, device):
        if self._gpu is None:
            self._gpu = tuple(torch.from_numpy(a).to(device) for a in
                              (self.src_req, self.src_j, self.block_table, self.seqused_k))
        return self._gpu


def plan_long(query_start_loc_cpu, seq_lens_cpu, block_size):
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
    src_req = np.repeat(np.arange(num_reqs), pages).astype(np.int32)
    src_j = (np.arange(total) - np.repeat(off, pages)).astype(np.int32)
    # FlashAttention looks up a page for every row of a KV tile (up to 256 tokens), also past the last
    # key of the request (those loads are masked, the page-table reads are not): keep the table at least
    # one tile wider than the longest request; the extra entries point at temporary page 0.
    bt = np.zeros((num_reqs, int(pages.max()) + -(-256 // block_size)), dtype=np.int32)
    bt[src_req, src_j] = np.arange(total)
    # Short requests get as many keys as query tokens (temporary page 0; their output is overwritten by
    # the INT8 kernel). Zero keys would be cheaper, but FlashAttention-2 writes out of bounds for such rows.
    return LongPlan(src_req=src_req, src_j=src_j, block_table=bt,
                    seqused_k=np.where(long, seq, q_lens).astype(np.int32), max_seqlen_k=int(seq[long].max()),
                    has_short=bool((q_lens[~long] > 0).any()), block_size=block_size)


def long_attention(q, k_cache, v_cache, out, cu_seqlens_q, block_table, max_query_len, plan, softmax_scale, static):
    """Dequantize the long requests' pages, then FlashAttention over the whole batch (short requests
    read a few keys of temporary page 0 there and are overwritten by decode_attention afterwards)."""
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, get_flash_attn_version

    bs, h, d = k_cache.shape[1:]
    assert bs == plan.block_size, (bs, plan.block_size)
    src_req, src_j, tmp_bt, seqused_k = plan.gpu(q.device)
    kt = torch.empty(src_req.shape[0], bs, h, d, dtype=q.dtype, device=q.device)
    vt = torch.empty_like(kt)
    ext().dequant_pages(k_cache, v_cache, kt, vt, block_table, src_req, src_j, static["k_scale"], static["v_scale"])
    if os.environ.get("KVQ_DEBUG_DUMP"):  # metadata of the last FlashAttention call, for an offline replay
        torch.save({"cu_seqlens_q": cu_seqlens_q.cpu(), "seqused_k": seqused_k.cpu(), "block_table": tmp_bt.cpu(),
                    "max_seqlen_q": max_query_len, "max_seqlen_k": plan.max_seqlen_k, "q_shape": tuple(q.shape),
                    "q_stride": q.stride(), "out_shape": tuple(out.shape), "out_stride": out.stride(),
                    "kt_shape": tuple(kt.shape)}, os.environ["KVQ_DEBUG_DUMP"])
    flash_attn_varlen_func(q=q, k=kt, v=vt, out=out, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_query_len,
                           seqused_k=seqused_k, max_seqlen_k=plan.max_seqlen_k, softmax_scale=softmax_scale,
                           causal=True, block_table=tmp_bt, fa_version=get_flash_attn_version())
    return out


def check_inputs(q, k_cache, cu_seqlens_q, seq_lens, block_table, plan):
    """KVQ_DEBUG=1: host-side bounds checks of one batch before the kernels run (syncs the GPU;
    skipped while a CUDA graph is being captured)."""
    if torch.cuda.is_current_stream_capturing():
        return
    cu, sl, bt = cu_seqlens_q.cpu().long(), seq_lens.cpu().long(), block_table.cpu().long()
    nb, bs = k_cache.shape[:2]
    n = cu.numel() - 1
    q_lens = cu[1:] - cu[:-1]
    errs = []
    if int(cu[-1]) > q.shape[0]:
        errs.append(f"cu_seqlens_q[-1] {int(cu[-1])} > {q.shape[0]} query rows")
    if sl.numel() < n or bt.shape[0] < n:
        errs.append(f"seq_lens {sl.numel()} / block_table {tuple(bt.shape)} rows < {n} requests")
    for r in torch.nonzero(q_lens > 0).flatten().tolist():
        L, ql = int(sl[r]), int(q_lens[r])
        pages = -(-L // bs)
        if L < ql:
            errs.append(f"req {r}: seq_len {L} < q_len {ql}")
        if pages > bt.shape[1]:
            errs.append(f"req {r}: {pages} pages > block_table width {bt.shape[1]}")
        elif pages and (int(bt[r, :pages].min()) < 0 or int(bt[r, :pages].max()) >= nb):
            errs.append(f"req {r} (q_len {ql}, seq_len {L}): page ids {bt[r, :pages].tolist()[:8]}... outside [0, {nb})")
    if plan is not None:
        su = torch.from_numpy(plan.seqused_k).long()
        sp = bt[torch.from_numpy(plan.src_req).long(), torch.from_numpy(plan.src_j).long()]
        if sp.numel() and (int(sp.min()) < 0 or int(sp.max()) >= nb):
            errs.append(f"pages to dequantize outside [0, {nb})")
        if su.numel() != n:
            errs.append(f"seqused_k has {su.numel()} rows, batch has {n}")
        else:
            long = q_lens > SHORT_Q
            diff = torch.nonzero(long & (su != sl[:n])).flatten().tolist()
            if diff:
                r = diff[0]
                errs.append(f"{len(diff)} long requests with CPU seq_len != GPU seq_len, e.g. req {r}: "
                            f"{int(su[r])} vs {int(sl[r])} (q_len {int(q_lens[r])})")
    if errs:
        raise RuntimeError("kvq input check: " + "; ".join(errs[:8]))


def paged_attention(q, k_cache, v_cache, out, cu_seqlens_q, seq_lens, block_table, max_query_len, softmax_scale,
                    static, plan=None, segm=None):
    """Both paths for one batch, as the backend runs them (plan from plan_long, None if all short)."""
    if os.environ.get("KVQ_DEBUG") == "1":
        check_inputs(q, k_cache, cu_seqlens_q, seq_lens, block_table, plan)
    if plan is not None:
        long_attention(q, k_cache, v_cache, out, cu_seqlens_q, block_table, max_query_len, plan, softmax_scale,
                       static)
    if plan is None or plan.has_short:
        decode_attention(q, k_cache, v_cache, out, cu_seqlens_q, seq_lens, block_table, softmax_scale, static,
                         segm, max_query_len)
    return out
