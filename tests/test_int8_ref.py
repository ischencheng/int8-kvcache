"""Stage 4 unit tests: quantization bounds and the algebra of implementation B."""
import pytest
import torch
from conftest import random_paged_case, record

from kvq import quant
from kvq.int8_ref import CONFIGS, Cfg, attend_impl_b, paged_int8_attention, quantize_kv
from kvq.ref_attn import compare, paged_attention_ref


@pytest.mark.parametrize("gran", quant.GRANULARITIES)
def test_roundtrip_error_bound(gran):
    x = torch.randn(512, 4, 128, device="cuda") * torch.linspace(0.1, 30, 128, device="cuda")
    y, s = quant.quant_dequant(x, gran)
    assert ((x - y).abs() <= s / 2 + 1e-6).all()


@pytest.mark.parametrize("k_gran", ["per_head", "per_channel", "per_token_head", "per_tensor"])
def test_impl_b_algebra_is_exact(k_gran):
    """With q and P left unquantized, B must equal attention on the dequantized K/V
    computed in float64: this checks the scale folding / factoring, not rounding."""
    g = torch.Generator(device="cuda").manual_seed(0)
    b, R, K, hkv, G, D = 3, 2, 70, 2, 3, 64
    qb = torch.randn(b, R, hkv * G, D, generator=g, device="cuda")
    k = torch.randn(b, K, hkv, D, generator=g, device="cuda") * torch.linspace(0.1, 20, D, device="cuda")
    v = torch.randn(b, K, hkv, D, generator=g, device="cuda")
    valid = torch.ones(b, K, dtype=torch.bool, device="cuda")
    mask = valid[:, None].expand(b, R, K)
    cfg = Cfg(k_gran=k_gran, v_gran="per_channel", q_gran="none", p_quant="none")
    out = attend_impl_b(qb, k, v, mask, valid, 0.125, cfg)
    kq, ks, vq, vs = quantize_kv(k, v, valid, cfg)
    kd, vd = kq.double() * ks.double(), vq.double() * vs.double()
    qg = qb.double().reshape(b, R, hkv, G, D)
    p = torch.softmax(torch.einsum("brhgd,bkhd->bhgrk", qg, kd) * 0.125, -1)
    ref = torch.einsum("bhgrk,bkhd->brhgd", p, vd).flatten(2, 3)
    assert compare(ref, out)["rel_l2"] < 1e-6


@pytest.mark.parametrize("name", list(CONFIGS))
def test_random_paged_error(name):
    q, k, v, cu, sl, bt = random_paged_case(28, 4, [1, 1, 7, 33, 1], [600, 1500, 7, 500, 31])
    ref = paged_attention_ref(q, k, v, cu, sl, bt, 128 ** -0.5)
    err = compare(ref, paged_int8_attention(CONFIGS[name])(q, k, v, cu, sl, bt, 128 ** -0.5))
    record("int8_ref_random", config=name, **err)
    assert err["rel_l2"] < 0.05
