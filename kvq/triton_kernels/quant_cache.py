"""Fused "quantize + reshape_and_cache": one kernel reads the new K/V once and writes int8.

Without fusion the static path is two passes over the new tokens (quantize into a
temporary, then scatter into the paged cache). Here each program takes one token, all
its kv heads, subtracts the smooth-K mean, divides by the static scale, rounds half
to even (same as torch.round) and stores into the slot the scheduler assigned.
With static=None it is a plain bf16 reshape_and_cache (the same-kernel baseline).
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _quant_cache_kernel(
    key, value, k_cache, v_cache, slots, k_scale, v_scale, k_mu,
    stride_kt, stride_kh, stride_vt, stride_vh,
    stride_kcb, stride_kcn, stride_kch, stride_vcb, stride_vcn, stride_vch,
    H: tl.constexpr, H_PAD: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    QUANT: tl.constexpr, K_PER_CHANNEL: tl.constexpr, V_PER_CHANNEL: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(slots + t).to(tl.int64)
    if slot < 0:  # padding token
        return
    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE
    h = tl.arange(0, H_PAD)
    d = tl.arange(0, D)
    hm = (h < H)[:, None]
    k = tl.load(key + t * stride_kt + h[:, None] * stride_kh + d[None, :], mask=hm)
    v = tl.load(value + t * stride_vt + h[:, None] * stride_vh + d[None, :], mask=hm)
    if QUANT:
        hd = h[:, None] * D + d[None, :]
        mu = tl.load(k_mu + hd, mask=hm, other=0.0)
        if K_PER_CHANNEL:
            ks = tl.load(k_scale + hd, mask=hm, other=1.0)
        else:
            ks = tl.load(k_scale + h, mask=h < H, other=1.0)[:, None]
        if V_PER_CHANNEL:
            vs = tl.load(v_scale + hd, mask=hm, other=1.0)
        else:
            vs = tl.load(v_scale + h, mask=h < H, other=1.0)[:, None]
        kq = libdevice.rint(tl.div_rn(k.to(tl.float32) - mu, ks))
        vq = libdevice.rint(tl.div_rn(v.to(tl.float32), vs))
        k = tl.minimum(tl.maximum(kq, -127.0), 127.0).to(tl.int8)
        v = tl.minimum(tl.maximum(vq, -127.0), 127.0).to(tl.int8)
    tl.store(k_cache + blk * stride_kcb + off * stride_kcn + h[:, None] * stride_kch + d[None, :], k, mask=hm)
    tl.store(v_cache + blk * stride_vcb + off * stride_vcn + h[:, None] * stride_vch + d[None, :], v, mask=hm)


def quant_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, static=None):
    """key/value [T, H, D] (T may exceed len(slot_mapping): padding); caches [blocks, block, H, D] views."""
    n = slot_mapping.shape[0]
    if n == 0:
        return
    H, D = key.shape[1], key.shape[2]
    assert key.stride(-1) == 1 and key_cache.stride(-1) == 1
    quant = static is not None
    ks = static["k_scale"] if quant else key
    vs = static["v_scale"] if quant else key
    mu = static["k_mu"] if quant else key
    _quant_cache_kernel[(n,)](
        key, value, key_cache, value_cache, slot_mapping, ks, vs, mu,
        key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        H=H, H_PAD=triton.next_power_of_2(H), D=D, BLOCK_SIZE=key_cache.shape[1],
        QUANT=quant, K_PER_CHANNEL=quant and ks.shape[-1] > 1, V_PER_CHANNEL=quant and vs.shape[-1] > 1,
        num_warps=4,
    )
