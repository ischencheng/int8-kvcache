"""MLA references: INT8 latent decode vs bf16, and the smooth (mean) add-back in the PV role."""
import torch

from kvq import mla, quant
from kvq.ref_attn import compare


def case(offset=0.0, seed=0, B=3, N=16, T=300, bs=64):
    g = torch.Generator().manual_seed(seed)
    pages = (T + bs - 1) // bs
    cache = torch.randn(B * pages + 2, bs, 576, generator=g)
    cache[..., :8] += offset                      # a few latent channels with a large constant offset
    bt = torch.arange(B * pages, dtype=torch.int32).view(B, pages) + 1
    sl = torch.tensor([T, T // 2, 7], dtype=torch.int32)
    q_lat = torch.randn(B, N, 512, generator=g) * 0.1
    q_pe = torch.randn(B, N, 64, generator=g) * 0.1
    return q_lat, q_pe, cache, bt, sl


def static_for(cache, smooth):
    c = cache[..., :512].flatten(0, 1)
    mu = c.mean(0) if smooth else torch.zeros(512)
    return {"c_mu": mu, "c_scale": quant.scale_from_absmax((c - mu).abs().amax(0)),
            "kpe_scale": quant.scale_from_absmax(cache[..., 512:].flatten(0, 1).abs().amax(0))}


def test_int8_latent_close_to_bf16():
    q_lat, q_pe, cache, bt, sl = case()
    ref = mla.decode_ref(q_lat, q_pe, cache, bt, sl, 0.1)
    for impl in ("A", "B"):
        out = mla.decode_ref(q_lat, q_pe, cache, bt, sl, 0.1, {"impl": impl, "smooth": True, "rope_int8": True},
                             static_for(cache, True))
        assert compare(ref, out)["rel_l2"] < 0.02, impl


def test_smooth_needs_mean_added_back_in_pv():
    """With constant offsets in some latent channels, smoothing must help, and the output must
    still contain the offset (it is a V-side quantity): dropping + mu would be wrong."""
    q_lat, q_pe, cache, bt, sl = case(offset=40.0)
    ref = mla.decode_ref(q_lat, q_pe, cache, bt, sl, 0.1)
    plain = mla.decode_ref(q_lat, q_pe, cache, bt, sl, 0.1, {"impl": "B", "smooth": False, "rope_int8": False},
                           static_for(cache, False))
    st = static_for(cache, True)
    smooth = mla.decode_ref(q_lat, q_pe, cache, bt, sl, 0.1, {"impl": "B", "smooth": True, "rope_int8": False}, st)
    e_plain, e_smooth = compare(ref, plain)["rel_l2"], compare(ref, smooth)["rel_l2"]
    assert e_smooth < e_plain
    assert compare(ref, smooth - st["c_mu"])["rel_l2"] > 0.5      # without the add-back it is badly off
