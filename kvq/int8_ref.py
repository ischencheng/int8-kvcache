"""Stage 4: INT8 KV attention in PyTorch, two ways, on the stage-3 skeleton.

Implementation A: quantize K/V, dequantize right away, run the bf16 reference.
    Only the INT8 rounding of K/V differs from bf16, so its error is the
    quantization error alone. It is also exactly what a "dequantize on load"
    kernel (or chunked prefill reading an INT8 cache) computes.

Implementation B: real INT8 arithmetic, as an INT8 tensor-core kernel would do.
    QK: a scale can leave the dot product only if it does not depend on the
        contracted dim (head_dim). per_head / per_token_head K scales do not, so
        scores = (q_i8 . k_i8) * s_q * s_k. per_channel K scales do, so they are
        folded into q first (q' = q * s_k[h, :]) and q' is quantized.
    PV: contracted over tokens. per_channel / per_head V scales do not depend on
        the token, so o = (p . v_i8) * s_v. P itself is quantized per row and per
        KV tile (scale = tile max / 127), like a flash kernel that quantizes each
        tile of exp(s - m) before its INT8 PV mma.
    Integer dot products are evaluated in float64, where they are exact.

Scales come from the K/V of each request (dynamic, stage 5) or from a calibration
file (static, stage 6). With smooth_k, a per-channel mean mu is removed from K
before quantization; softmax ignores the constant q.mu so nothing is added back.
"""
from dataclasses import dataclass, replace

import torch

from kvq import quant
from kvq.ref_attn import attend_bf16, group_heads, masked_softmax, pv, run_paged

_DIMS = {"per_tensor": (1, 2, 3), "per_head": (1, 3), "per_channel": (1,), "per_token_head": (3,)}


@dataclass(frozen=True)
class Cfg:
    impl: str = "B"                 # "A" (quant-dequant + bf16) or "B" (INT8 arithmetic)
    k_gran: str = "per_channel"     # per_tensor | per_head | per_channel | per_token_head
    v_gran: str = "per_channel"
    smooth_k: bool = False
    q_gran: str = "token_head"      # B only: none | tensor | token_head | group<n> (n channels per scale)
    p_quant: str = "tile"           # B only: none | static | tile
    p_tile: int = 64
    p_norm: str = "quant"           # B only: normalize by the sum of quantized (quant) or exact (exact) P


def request_absmax(x, valid, gran):
    """Dynamic |x| max per request over its valid tokens. x [b, K, H, D], valid [b, K]."""
    a = x.abs().float().masked_fill(~valid[:, :, None, None], 0)
    return a.amax(dim=_DIMS[gran], keepdim=True)


def request_mean(x, valid):
    w = valid[:, :, None, None].float()
    return (x.float() * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1)


def quantize_kv(k, v, valid, cfg, static=None):
    """Returns (k_i8, k_scale, v_i8, v_scale) with scales shaped to broadcast on [b, K, H, D].

    static: optional dict with k_scale/v_scale [H, D] (or [H, 1]) and k_mu [H, D] from calibration.
    """
    if cfg.smooth_k:
        mu = static["k_mu"][None, None] if static else request_mean(k, valid)
        k = k.float() - mu
    if static:
        ks, vs = static["k_scale"][None, None], static["v_scale"][None, None]
    else:
        ks = quant.scale_from_absmax(request_absmax(k, valid, cfg.k_gran))
        vs = quant.scale_from_absmax(request_absmax(v, valid, cfg.v_gran))
    return quant.quantize(k, ks), ks, quant.quantize(v, vs), vs


def attend_dequant(qb, kq, ks, vq, vs, mask, softmax_scale):
    """Implementation A on already-quantized K/V: dequantize (fp32 -> bf16) and run bf16 attention."""
    return attend_bf16(qb, quant.dequantize(kq, ks, torch.bfloat16), quant.dequantize(vq, vs, torch.bfloat16),
                       mask, softmax_scale)


def attend_impl_a(qb, k, v, mask, valid, softmax_scale, cfg, static=None):
    kq, ks, vq, vs = quantize_kv(k, v, valid, cfg, static)
    return attend_dequant(qb, kq, ks, vq, vs, mask, softmax_scale)


def _quant_q(qf, q_gran):
    if q_gran == "none":
        return qf, torch.ones_like(qf[..., :1])
    if q_gran == "tensor":
        s = quant.scale_from_absmax(qf.abs().amax())
        s = s.expand_as(qf[..., :1])
    else:  # one scale per (token, q head): still constant along head_dim
        s = quant.scale_from_absmax(qf.abs().amax(-1, keepdim=True))
    return quant.quantize(qf, s).double(), s


def _quant_p(e, cfg):
    """e = exp(s - rowmax) in [0, 1], [b, Hkv, G, R, K]. Returns its dequantized value."""
    if cfg.p_quant == "none":
        return e
    if cfg.p_quant == "static":
        return torch.round(e * quant.QMAX) / quant.QMAX
    K = e.shape[-1]
    pad = (-K) % cfg.p_tile
    t = torch.nn.functional.pad(e, (0, pad)).unflatten(-1, (-1, cfg.p_tile))
    s = (t.amax(-1, keepdim=True) / quant.QMAX).clamp(min=1e-30)
    t = torch.round(t / s) * s
    return t.flatten(-2)[..., :K]


def attend_int8(qb, kq, ks, vq, vs, mask, softmax_scale, cfg):
    """Implementation B on already-quantized K/V (int8 values + broadcastable scales)."""
    assert cfg.v_gran != "per_token_head", "a per-token V scale sits on the PV contraction dim"
    hkv = kq.shape[2]
    qg = group_heads(qb, hkv).double()                       # [b, R, Hkv, G, D]
    # --- QK ---
    if ks.shape[-1] > 1:                                     # per_channel: fold s_k[h, d] into q
        qf = qg * ks.double()[:, :, :, None, :]
        k_post = 1.0
    else:
        qf = qg
        k_post = ks.double().permute(0, 2, 3, 1).unsqueeze(2)   # [b, Hkv|1, 1, 1, K|1] on the scores
    if cfg.q_gran.startswith("group"):
        # One q scale per slice of head_dim: each slice's integer dot is scaled
        # on its own and the slices are summed in fp32 (still exact algebra).
        n = int(cfg.q_gran[5:])
        qs_ = qf.unflatten(-1, (-1, n))                       # [b, R, Hkv, G, D/n, n]
        q_s = quant.scale_from_absmax(qs_.abs().amax(-1, keepdim=True))
        q_i = quant.quantize(qs_, q_s).double()
        s_int = torch.einsum("brhgcn,bkhcn->bhgrkc", q_i, kq.double().unflatten(-1, (-1, n)))
        scores = (s_int * q_s.double()[..., 0].permute(0, 2, 3, 1, 4)[:, :, :, :, None, :]).sum(-1)
        scores = scores * k_post * softmax_scale
    else:
        q_i, q_s = _quant_q(qf, cfg.q_gran)                   # q_s [b, R, Hkv, G, 1]
        s_int = torch.einsum("brhgd,bkhd->bhgrk", q_i, kq.double())
        scores = s_int * q_s.double().permute(0, 2, 3, 1, 4) * k_post * softmax_scale
    # --- softmax in fp32 terms, P quantized per (row, tile) ---
    scores = scores.float().masked_fill(~mask[:, None, None], float("-inf"))
    e = torch.exp(scores - scores.amax(-1, keepdim=True))
    e_hat = _quant_p(e, cfg)
    denom = (e_hat if cfg.p_norm == "quant" else e).sum(-1, keepdim=True)
    # --- PV ---
    o = torch.einsum("bhgrk,bkhd->brhgd", e_hat.double(), vq.double())
    o = o / denom.double().permute(0, 3, 1, 2, 4)
    o = o * vs.double()[:, :, :, None, :]                    # s_v[h, d] is constant over tokens
    return o.flatten(2, 3).float()


def attend_impl_b(qb, k, v, mask, valid, softmax_scale, cfg, static=None):
    kq, ks, vq, vs = quantize_kv(k, v, valid, cfg, static)
    return attend_int8(qb, kq, ks, vq, vs, mask, softmax_scale, cfg)


def paged_int8_attention(cfg, static=None):
    """A drop-in paged attention function (same signature as paged_attention_ref)."""
    attend = attend_impl_a if cfg.impl == "A" else attend_impl_b

    def fn(q, k_cache, v_cache, cu_seqlens_q, seq_lens, block_table, softmax_scale, causal=True):
        return run_paged(lambda qb, k, v, mask, valid: attend(qb, k, v, mask, valid, softmax_scale, cfg, static),
                         q, k_cache, v_cache, cu_seqlens_q, seq_lens, block_table, causal)

    return fn


def paged_static_attention(cfg, static, cache_is_int8):
    """Static-scale attention as the INT8 backend runs it: decode rows use the INT8
    arithmetic (B), multi-token rows (prefill / chunked prefill) dequantize (A).

    cache_is_int8=True : the cache already holds int8 (the real static backend).
    cache_is_int8=False: the cache is bf16 and is quantized with the static scales at
                         read time (the stage-6 bisect tool: same math, no framework change).
    """
    ks, vs = static["k_scale"][None, None], static["v_scale"][None, None]
    mu = static.get("k_mu")

    def prep(k, v):
        if cache_is_int8:
            return k, v
        kf = k.float() - (mu[None, None] if mu is not None else 0)
        return quant.quantize(kf, ks), quant.quantize(v, vs)

    def decode(qb, k, v, mask, valid):
        kq, vq = prep(k, v)
        return attend_int8(qb, kq, ks, vq, vs, mask, softmax_scale_[0], cfg)

    def prefill(qb, k, v, mask, valid):
        kq, vq = prep(k, v)
        return attend_dequant(qb, kq, ks, vq, vs, mask, softmax_scale_[0])

    softmax_scale_ = [None]

    def fn(q, k_cache, v_cache, cu_seqlens_q, seq_lens, block_table, softmax_scale, causal=True):
        softmax_scale_[0] = softmax_scale
        return run_paged(decode, q, k_cache, v_cache, cu_seqlens_q, seq_lens, block_table, causal,
                         attend_multi=prefill)

    return fn


# Named configurations used by tests, reports and the stage-5 vLLM override (KVQ_ATTN=<name>).
CONFIGS = {
    "A_head": Cfg(impl="A", k_gran="per_head", v_gran="per_head"),
    "A_channel": Cfg(impl="A"),
    "A_channel_smooth": Cfg(impl="A", smooth_k=True),
    "A_token_head": Cfg(impl="A", k_gran="per_token_head", v_gran="per_token_head"),
    "B_head": Cfg(k_gran="per_head", v_gran="per_head"),
    "B_channel": Cfg(),
    "B_channel_smooth": Cfg(smooth_k=True),
    # ablations of implementation B
    "B_channel_qtensor": Cfg(q_gran="tensor"),
    "B_channel_pnone": Cfg(p_quant="none"),
    "B_channel_pstatic": Cfg(p_quant="static"),
    "B_channel_pexactnorm": Cfg(p_norm="exact"),
    "A_head_smooth": Cfg(impl="A", k_gran="per_head", v_gran="per_head", smooth_k=True),
    "A_token_head_smooth": Cfg(impl="A", k_gran="per_token_head", v_gran="per_token_head", smooth_k=True),
    "B_head_smooth": Cfg(k_gran="per_head", v_gran="per_head", smooth_k=True),
    "B_channel_smooth_pnone": Cfg(smooth_k=True, p_quant="none"),
    "B_channel_smooth_pstatic": Cfg(smooth_k=True, p_quant="static"),
    "B_channel_smooth_qnone": Cfg(smooth_k=True, q_gran="none"),
    "B_channel_smooth_qgroup32": Cfg(smooth_k=True, q_gran="group32"),
    "B_channel_qgroup32": Cfg(q_gran="group32"),
}
MODES = {name: paged_int8_attention(cfg) for name, cfg in CONFIGS.items()}


def with_(cfg, **kw):
    return replace(cfg, **kw)
