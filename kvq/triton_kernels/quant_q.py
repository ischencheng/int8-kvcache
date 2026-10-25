"""q quantization for the INT8 QK product (the "q 的量化算子" of stage 7).

Per (token, q head): fold the per-channel K scale of its kv head into q (q' = q * s_k[h // G, :]),
then quantize q' symmetrically with one scale per GROUP channels (GROUP = D: one per row).
The attention kernel can also do this in its prologue (fused); this standalone kernel is
the unfused variant and a unit-test target.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _quant_q_kernel(q, q_i8, q_scale, k_scale, stride_qt, stride_qh,
                    G: tl.constexpr, D: tl.constexpr, GROUP: tl.constexpr, K_PER_CHANNEL: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    NG: tl.constexpr = D // GROUP
    g = tl.arange(0, NG)
    c = tl.arange(0, GROUP)
    d = g[:, None] * GROUP + c[None, :]                        # [NG, GROUP] channel ids
    x = tl.load(q + t * stride_qt + h * stride_qh + d).to(tl.float32)
    if K_PER_CHANNEL:
        x = x * tl.load(k_scale + (h // G) * D + d)
    s = tl.maximum(tl.max(tl.abs(x), axis=1) / 127.0, 1e-8)    # [NG]
    xq = libdevice.rint(tl.div_rn(x, s[:, None]))
    xq = tl.minimum(tl.maximum(xq, -127.0), 127.0).to(tl.int8)
    row = t * (tl.num_programs(1) * D) + h * D
    tl.store(q_i8 + row + d, xq)
    tl.store(q_scale + (t * tl.num_programs(1) + h) * NG + g, s)


def quant_q(q, k_scale, group=None):
    """q [T, Hq, D] -> (q_i8 [T, Hq, D], q_scale [T, Hq, D // group] fp32)."""
    T, Hq, D = q.shape
    group = group or D
    G = Hq // k_scale.shape[0]
    q_i8 = torch.empty(T, Hq, D, dtype=torch.int8, device=q.device)
    q_scale = torch.empty(T, Hq, D // group, dtype=torch.float32, device=q.device)
    if T:
        _quant_q_kernel[(T, Hq)](q, q_i8, q_scale, k_scale, q.stride(0), q.stride(1), G=G, D=D, GROUP=group,
                                 K_PER_CHANNEL=k_scale.shape[-1] > 1, num_warps=1)
    return q_i8, q_scale
