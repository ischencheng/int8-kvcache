"""Stage 7: paged attention over an INT8 (or bf16) KV cache, one Triton kernel for
prefill, chunked prefill, decode and spec-decode verification.

Launch layout (same idea as vLLM's unified Triton attention): a program owns BLOCK_M
rows = BLOCK_Q query tokens x G q heads of one kv head (GQA: the G heads share every
K/V tile it loads), walks the request's pages through the block table in tiles of
TILE_N tokens and keeps an online softmax. Pure-decode batches with few requests split
the KV range into NUM_SEGMENTS programs whose partial results a second kernel merges
(flash-decoding), so short batches still fill the GPU.

MODE (constexpr, picked per launch):
  0  bf16 cache, bf16 math                        (same-kernel baseline)
  1  int8 cache, dequantize each tile to bf16      (implementation A: prefill / chunked prefill)
  2  int8 cache, INT8 tensor-core math            (implementation B: decode)
       prologue: q' = q * s_k[kvh, :] (per-channel K scale folded in), quantize q' per row
                 or per 32-channel group (Q_GROUPS = 4)
       QK:  int32 = q_i8 . k_i8^T, times the q scale(s)
       PV:  P = exp2(s - m) quantized per (row, tile): p_i8 = rint(P * 127 / rowmax), then
            int32 = p_i8 . v_i8, times rowmax / 127; s_v (per channel) applied at the end
Smooth-K needs nothing here: the cache holds k - mu and softmax ignores q.mu.
"""
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

LOG2E = 1.4426950408889634


@triton.jit
def _find_seq(cu_q, pid, num_seqs, BLOCK_Q: tl.constexpr):
    """Largest i with cu_q[i] // BLOCK_Q + i <= pid (each request's q blocks start there)."""
    lo = 0
    hi = num_seqs
    while lo < hi:
        mid = (lo + hi) // 2
        if tl.load(cu_q + mid) // BLOCK_Q + mid <= pid:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


@triton.jit
def _quant_rows(x):
    """Symmetric int8 per row of a [M, n] fp32 tile -> (int8 tile, fp32 scale [M])."""
    s = tl.maximum(tl.max(tl.abs(x), axis=1) / 127.0, 1e-8)
    xq = libdevice.rint(x / s[:, None])
    return tl.minimum(tl.maximum(xq, -127.0), 127.0).to(tl.int8), s


@triton.jit
def _attn_kernel(
    q_ptr, q_i8_ptr, q_s_ptr, k_ptr, v_ptr, out_ptr,
    segm_out, segm_max, segm_sum,
    cu_q, seq_lens, block_table,
    k_scale, v_scale,
    sm_scale, num_seqs,
    stride_qt, stride_qh, stride_ot, stride_oh,
    stride_kb, stride_kn, stride_kh, stride_vb, stride_vn, stride_vh,
    stride_bt, stride_st, stride_sh,
    G: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_Q: tl.constexpr, TILE_N: tl.constexpr,
    MODE: tl.constexpr, K_PER_CHANNEL: tl.constexpr, V_PER_CHANNEL: tl.constexpr,
    Q_GROUPS: tl.constexpr, Q_PREQUANT: tl.constexpr, NUM_SEGMENTS: tl.constexpr,
):
    pid_q = tl.program_id(0)
    kvh = tl.program_id(1)
    seg = tl.program_id(2)
    seq = _find_seq(cu_q, pid_q, num_seqs, BLOCK_Q)
    q_start = tl.load(cu_q + seq)
    q_len = tl.load(cu_q + seq + 1) - q_start
    qblk = pid_q - (q_start // BLOCK_Q + seq)
    if qblk * BLOCK_Q >= q_len:
        return
    seq_len = tl.load(seq_lens + seq)
    ctx_len = seq_len - q_len

    rows = tl.arange(0, BLOCK_M)
    tok = qblk * BLOCK_Q + rows // G                  # token index inside the request
    head = kvh * G + rows % G
    row_ok = (rows < BLOCK_Q * G) & (tok < q_len)
    qrow = (q_start + tok).to(tl.int64)
    d = tl.arange(0, D)
    GW: tl.constexpr = D // 4                          # channel group width when Q_GROUPS == 4

    # ---- prologue: load (and for MODE 2 quantize) the q tile ----
    if MODE == 2:
        if K_PER_CHANNEL:
            ks_row = tl.load(k_scale + kvh * D + d)
            post = sm_scale
        else:
            ks_row = tl.full([D], 1.0, tl.float32)
            post = sm_scale * tl.load(k_scale + kvh)
        if Q_GROUPS == 1:
            if Q_PREQUANT:
                q0 = tl.load(q_i8_ptr + qrow[:, None] * (G * D * tl.num_programs(1)) + head[:, None] * D + d[None, :],
                             mask=row_ok[:, None], other=0)
                s0 = tl.load(q_s_ptr + qrow * (G * tl.num_programs(1)) + head, mask=row_ok, other=0.0)
            else:
                qf = tl.load(q_ptr + qrow[:, None] * stride_qt + head[:, None] * stride_qh + d[None, :],
                             mask=row_ok[:, None], other=0.0).to(tl.float32)
                q0, s0 = _quant_rows(qf * ks_row[None, :])
            s0 = s0 * post
        else:
            dg = tl.arange(0, GW)
            qb = q_ptr + qrow[:, None] * stride_qt + head[:, None] * stride_qh + dg[None, :]
            q0, s0 = _quant_rows(tl.load(qb, mask=row_ok[:, None], other=0.0).to(tl.float32)
                                 * tl.load(k_scale + kvh * D + dg)[None, :])
            q1, s1 = _quant_rows(tl.load(qb + GW, mask=row_ok[:, None], other=0.0).to(tl.float32)
                                 * tl.load(k_scale + kvh * D + GW + dg)[None, :])
            q2, s2 = _quant_rows(tl.load(qb + 2 * GW, mask=row_ok[:, None], other=0.0).to(tl.float32)
                                 * tl.load(k_scale + kvh * D + 2 * GW + dg)[None, :])
            q3, s3 = _quant_rows(tl.load(qb + 3 * GW, mask=row_ok[:, None], other=0.0).to(tl.float32)
                                 * tl.load(k_scale + kvh * D + 3 * GW + dg)[None, :])
            s0 = s0 * post
            s1 = s1 * post
            s2 = s2 * post
            s3 = s3 * post
    else:
        q = tl.load(q_ptr + qrow[:, None] * stride_qt + head[:, None] * stride_qh + d[None, :],
                    mask=row_ok[:, None], other=0.0)
        if MODE == 1:
            if K_PER_CHANNEL:
                ks_row = tl.load(k_scale + kvh * D + d)
            else:
                ks_row = tl.full([D], 1.0, tl.float32) * tl.load(k_scale + kvh)
    if MODE != 0:
        if V_PER_CHANNEL:
            vs_row = tl.load(v_scale + kvh * D + d)
        else:
            vs_row = tl.full([D], 1.0, tl.float32) * tl.load(v_scale + kvh)

    # ---- KV range of this program (causal: up to the last query of the block) ----
    q_last = tl.minimum(qblk * BLOCK_Q + BLOCK_Q, q_len) - 1
    kv_end = tl.minimum(seq_len, ctx_len + q_last + 1)
    num_tiles = tl.cdiv(kv_end, TILE_N)
    per_seg = tl.cdiv(num_tiles, NUM_SEGMENTS)
    t_begin = seg * per_seg
    t_end = tl.minimum(t_begin + per_seg, num_tiles)
    qpos = ctx_len + tok

    m = tl.full([BLOCK_M], -1e30, tl.float32)
    l = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    for t in range(t_begin, t_end):
        n = t * TILE_N + tl.arange(0, TILE_N)
        n_ok = n < kv_end
        blk = tl.load(block_table + seq * stride_bt + n // BLOCK_SIZE, mask=n_ok, other=0).to(tl.int64)
        off = n % BLOCK_SIZE
        k_base = k_ptr + blk[:, None] * stride_kb + off[:, None] * stride_kn + kvh * stride_kh
        v_base = v_ptr + blk[:, None] * stride_vb + off[:, None] * stride_vn + kvh * stride_vh
        if MODE == 2:
            if Q_GROUPS == 1:
                k = tl.load(k_base + d[None, :], mask=n_ok[:, None], other=0)
                s = tl.dot(q0, tl.trans(k), out_dtype=tl.int32).to(tl.float32) * s0[:, None]
            else:
                dg = tl.arange(0, GW)
                k0 = tl.load(k_base + dg[None, :], mask=n_ok[:, None], other=0)
                k1 = tl.load(k_base + GW + dg[None, :], mask=n_ok[:, None], other=0)
                k2 = tl.load(k_base + 2 * GW + dg[None, :], mask=n_ok[:, None], other=0)
                k3 = tl.load(k_base + 3 * GW + dg[None, :], mask=n_ok[:, None], other=0)
                s = tl.dot(q0, tl.trans(k0), out_dtype=tl.int32).to(tl.float32) * s0[:, None]
                s += tl.dot(q1, tl.trans(k1), out_dtype=tl.int32).to(tl.float32) * s1[:, None]
                s += tl.dot(q2, tl.trans(k2), out_dtype=tl.int32).to(tl.float32) * s2[:, None]
                s += tl.dot(q3, tl.trans(k3), out_dtype=tl.int32).to(tl.float32) * s3[:, None]
        else:
            k = tl.load(k_base + d[None, :], mask=n_ok[:, None], other=0)
            if MODE == 1:
                k = (k.to(tl.float32) * ks_row[None, :]).to(q.dtype)
            s = tl.dot(q, tl.trans(k)) * sm_scale
        vis = n_ok[None, :] & (n[None, :] <= qpos[:, None])
        s = tl.where(vis, s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp2(m - m_new)
        p = tl.exp2(s - m_new[:, None])
        v = tl.load(v_base + d[None, :], mask=n_ok[:, None], other=0)
        if MODE == 2:
            p_s = tl.maximum(tl.max(p, axis=1) / 127.0, 1e-30)
            p_i8 = libdevice.rint(p / p_s[:, None]).to(tl.int8)
            l = l * alpha + tl.sum(p_i8.to(tl.float32), axis=1) * p_s
            pv = tl.dot(p_i8, v, out_dtype=tl.int32).to(tl.float32)
            acc = acc * alpha[:, None] + pv * p_s[:, None]
        else:
            if MODE == 1:
                v = (v.to(tl.float32) * vs_row[None, :]).to(q.dtype)
            l = l * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new

    if MODE == 2:
        acc = acc * vs_row[None, :]
    if NUM_SEGMENTS == 1:
        o = acc / tl.maximum(l, 1e-30)[:, None]
        tl.store(out_ptr + qrow[:, None] * stride_ot + head[:, None] * stride_oh + d[None, :],
                 o.to(out_ptr.dtype.element_ty), mask=row_ok[:, None])
    else:
        sidx = qrow * stride_st + head * stride_sh + seg
        tl.store(segm_max + sidx, m, mask=row_ok)
        tl.store(segm_sum + sidx, l, mask=row_ok)
        tl.store(segm_out + sidx[:, None] * D + d[None, :], acc, mask=row_ok[:, None])


@triton.jit
def _reduce_kernel(out_ptr, segm_out, segm_max, segm_sum, stride_ot, stride_oh, stride_st, stride_sh,
                   D: tl.constexpr, NUM_SEGMENTS: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    s = tl.arange(0, NUM_SEGMENTS)
    d = tl.arange(0, D)
    idx = t * stride_st + h * stride_sh + s
    ms = tl.load(segm_max + idx)
    ls = tl.load(segm_sum + idx)
    big = tl.max(ms, axis=0)
    w = tl.exp2(ms - big)
    acc = tl.load(segm_out + idx[:, None] * D + d[None, :])
    o = tl.sum(acc * w[:, None], axis=0) / tl.maximum(tl.sum(ls * w, axis=0), 1e-30)
    tl.store(out_ptr + t * stride_ot + h * stride_oh + d, o.to(out_ptr.dtype.element_ty))


def paged_attention(q, k_cache, v_cache, out, cu_seqlens_q, seq_lens, block_table, max_query_len, softmax_scale,
                    static=None, int8_math=True, q_groups=1, q_prequant=None, segm=None, block_m=None, tile_n=None,
                    int8_max_q=1):
    """q/out [T, Hq, D]; caches [blocks, block, Hkv, D] views (int8 if static else bf16).

    static:     None (bf16) or dict of k_scale/v_scale ([Hkv, D] per channel or [Hkv, 1] per head).
    int8_math:  MODE 2 when every request has <= int8_max_q query tokens (decode; with int8_max_q > 1
                also spec-decode verification); MODE 1 (dequantize) otherwise.
    q_prequant: optional (q_i8, q_scale) from quant_q() to skip the fused q quantization.
    segm:       optional (out [Tmax, Hq, S, D], max [Tmax, Hq, S], sum) scratch; S > 1 enables
                split-KV for a pure-decode batch that fits in it.
    """
    T, Hq, D = q.shape
    Hkv = k_cache.shape[2]
    G = Hq // Hkv
    num_seqs = cu_seqlens_q.shape[0] - 1
    if T == 0:
        return out
    assert q_groups == 1 or (static is not None and static["k_scale"].shape[-1] > 1), "q groups need per-channel K"
    decode = max_query_len == 1
    if static is None:
        mode = 0
    else:
        mode = 2 if (max_query_len <= int8_max_q and int8_math) else 1
    block_m = block_m or (16 if max_query_len <= 8 else 64)
    block_m = max(block_m, triton.next_power_of_2(G))
    block_q = block_m // G
    tile_n = tile_n or 64
    use_segm = segm is not None and decode and T <= segm[1].shape[0] and segm[1].shape[2] > 1
    S = segm[1].shape[2] if use_segm else 1
    so, sm, ss = segm if use_segm else (out, out, out)
    dummy = q
    grid = (T // block_q + num_seqs, Hkv, S)
    ks = static["k_scale"] if static is not None else dummy
    vs = static["v_scale"] if static is not None else dummy
    qi, qs = q_prequant if q_prequant is not None else (dummy, dummy)
    _attn_kernel[grid](
        q, qi, qs, k_cache, v_cache, out, so, sm, ss,
        cu_seqlens_q, seq_lens, block_table, ks, vs,
        softmax_scale * LOG2E, num_seqs,
        q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_table.stride(0), sm.stride(0) if use_segm else 0, sm.stride(1) if use_segm else 0,
        G=G, D=D, BLOCK_SIZE=k_cache.shape[1], BLOCK_M=block_m, BLOCK_Q=block_q, TILE_N=tile_n,
        MODE=mode, K_PER_CHANNEL=static is not None and ks.shape[-1] > 1,
        V_PER_CHANNEL=static is not None and vs.shape[-1] > 1,
        Q_GROUPS=q_groups if mode == 2 else 1, Q_PREQUANT=q_prequant is not None and mode == 2,
        NUM_SEGMENTS=S, num_warps=4, num_stages=2,
    )
    if use_segm:
        _reduce_kernel[(T, Hq)](out, so, sm, ss, out.stride(0), out.stride(1), sm.stride(0), sm.stride(1),
                                D=D, NUM_SEGMENTS=S, num_warps=4)
    return out


INT8_MAX_Q = int(os.environ.get("KVQ_INT8_MAX_Q", "1"))


def int8_paged_attention(query, key_cache, value_cache, output, md, softmax_scale, static, cfg):
    """vLLM entry point: md is the Triton backend's metadata (reuses its split-KV scratch)."""
    num_reqs = md.query_start_loc.shape[0] - 1
    segm = None
    if md.max_query_len == 1 and num_reqs <= md.seq_threshold_3D:
        segm = (md.softmax_segm_output, md.softmax_segm_max, md.softmax_segm_expsum)
    q_groups = 4 if cfg.q_gran == "group32" else 1
    return paged_attention(query, key_cache, value_cache, output, md.query_start_loc, md.seq_lens,
                           md.block_table, md.max_query_len, softmax_scale, static,
                           int8_math=cfg.impl == "B", q_groups=q_groups, segm=segm, int8_max_q=INT8_MAX_Q)
