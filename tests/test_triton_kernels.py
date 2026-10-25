"""Stage 7: Triton kernels against the PyTorch references of stages 3/4."""
import pytest
import torch
from conftest import DUMPS, random_paged_case, record

from kvq import quant
from kvq.int8_ref import CONFIGS, attend_dequant, paged_static_attention
from kvq.ref_attn import compare, gather_pages, paged_attention_ref, run_paged
from kvq.triton_kernels.int8_attention import paged_attention
from kvq.triton_kernels.quant_cache import quant_reshape_and_cache
from kvq.triton_kernels.quant_q import quant_q

SCALE = 128 ** -0.5


def valid_rows(cache, block_table, seq_lens):
    """Only the slots requests actually wrote: the tail of a last page holds stale data."""
    return torch.cat([gather_pages(cache, block_table[i:i + 1], int(n))[0] for i, n in enumerate(seq_lens.tolist())])


def make_static(k_cache, v_cache, per_channel=True, smooth=True, block_table=None, seq_lens=None):
    """Static scales from the cache contents themselves (stand-in for a calibration)."""
    if block_table is not None:
        k, v = valid_rows(k_cache, block_table, seq_lens).float(), valid_rows(v_cache, block_table, seq_lens).float()
    else:
        k, v = k_cache.flatten(0, 1).float(), v_cache.flatten(0, 1).float()
    mu = k.mean(0) if smooth else torch.zeros_like(k[0])
    ka, va = (k - mu).abs().amax(0), v.abs().amax(0)
    if not per_channel:
        ka, va = ka.amax(-1, keepdim=True), va.amax(-1, keepdim=True)
    return {"k_scale": quant.scale_from_absmax(ka).contiguous(), "v_scale": quant.scale_from_absmax(va).contiguous(),
            "k_mu": mu.contiguous()}


def to_int8(k_cache, v_cache, st):
    return quant.quantize(k_cache.float() - st["k_mu"], st["k_scale"]), quant.quantize(v_cache, st["v_scale"])


@pytest.mark.parametrize("per_channel", [True, False])
def test_quant_cache_matches_torch(per_channel):
    g = torch.Generator(device="cuda").manual_seed(0)
    T, H, D, nb, bs = 200, 4, 128, 64, 16
    key = torch.randn(T + 8, H, D, generator=g, device="cuda").to(torch.bfloat16) * 5   # +8 padded rows
    value = torch.randn(T + 8, H, D, generator=g, device="cuda").to(torch.bfloat16)
    slots = torch.randperm(nb * bs, generator=g, device="cuda")[:T].long()
    slots[::17] = -1
    kv = torch.zeros(nb, H, bs, 2 * D, dtype=torch.int8, device="cuda")  # vLLM's logical layout
    kc, vc = kv.transpose(1, 2).split(D, dim=-1)
    st = make_static(key[:T, None], value[:T, None], per_channel)
    quant_reshape_and_cache(key, value, kc, vc, slots, st)
    ok = slots >= 0
    ref_k, ref_v = torch.zeros_like(kc), torch.zeros_like(vc)
    ref_k[slots[ok] // bs, slots[ok] % bs] = quant.quantize(key[:T][ok].float() - st["k_mu"], st["k_scale"])
    ref_v[slots[ok] // bs, slots[ok] % bs] = quant.quantize(value[:T][ok], st["v_scale"])
    assert torch.equal(kc, ref_k) and torch.equal(vc, ref_v)


@pytest.mark.parametrize("group", [128, 32])
def test_quant_q_matches_torch(group):
    g = torch.Generator(device="cuda").manual_seed(1)
    q = torch.randn(37, 28, 128, generator=g, device="cuda").to(torch.bfloat16)
    ks = torch.rand(4, 128, generator=g, device="cuda") + 0.01
    qi, qs = quant_q(q, ks, group)
    qf = (q.float().reshape(37, 4, 7, 128) * ks[None, :, None, :]).reshape(37, 28, 128 // group, group)
    s = quant.scale_from_absmax(qf.abs().amax(-1, keepdim=True))
    assert torch.equal(qi.reshape(37, 28, -1, group), quant.quantize(qf, s))
    assert torch.allclose(qs, s[..., 0])


CASES = {
    "decode": ([1] * 12, [17, 300, 1, 1024, 2047, 64, 129, 900, 3000, 5, 333, 1500]),
    "mixed": ([1, 1, 7, 33, 128, 1, 300], [600, 31, 7, 500, 128, 1500, 900]),
    "prefill": ([256, 97], [256, 1200]),
}


@pytest.mark.parametrize("heads", [(28, 4), (12, 2)])
@pytest.mark.parametrize("case", list(CASES))
def test_bf16_mode_matches_ref(heads, case):
    q, k, v, cu, sl, bt = random_paged_case(*heads, *CASES[case])
    out = paged_attention(q, k, v, torch.empty_like(q), cu, sl, bt, int((cu[1:] - cu[:-1]).max()), SCALE)
    err = compare(paged_attention_ref(q, k, v, cu, sl, bt, SCALE), out)
    record("triton_bf16", heads=heads, case=case, **err)
    assert err["rel_l2"] < 4e-3


@pytest.mark.parametrize("case", ["mixed", "prefill"])
def test_dequant_mode_matches_ref(case):
    q, k, v, cu, sl, bt = random_paged_case(28, 4, *CASES[case], scale=3.0)
    st = make_static(k, v)
    kq, vq = to_int8(k, v, st)
    ks, vs = st["k_scale"][None, None], st["v_scale"][None, None]
    ref = run_paged(lambda qb, kk, vv, mask, valid: attend_dequant(qb, kk, ks, vv, vs, mask, SCALE), q, kq, vq, cu, sl, bt)
    out = paged_attention(q, kq, vq, torch.empty_like(q), cu, sl, bt, int((cu[1:] - cu[:-1]).max()), SCALE, st)
    err = compare(ref, out)
    record("triton_dequant", case=case, **err)
    assert err["rel_l2"] < 4e-3


def _segm(T, hq, S=16, D=128):
    return (torch.empty(T, hq, S, D, device="cuda"), torch.empty(T, hq, S, device="cuda"),
            torch.empty(T, hq, S, device="cuda"))


@pytest.mark.parametrize("variant", ["fused_q", "prequant_q", "group32", "segments", "per_head"])
def test_int8_mode_matches_ref(variant):
    q, k, v, cu, sl, bt = random_paged_case(28, 4, *CASES["decode"], scale=3.0)
    st = make_static(k, v, per_channel=variant != "per_head")
    kq, vq = to_int8(k, v, st)
    cfg = CONFIGS["B_channel_smooth_qgroup32" if variant == "group32" else
                  "B_head_smooth" if variant == "per_head" else "B_channel_smooth"]
    ref = paged_static_attention(cfg, st, cache_is_int8=True)(q, kq, vq, cu, sl, bt, SCALE)
    kw = {}
    if variant == "prequant_q":
        kw["q_prequant"] = quant_q(q, st["k_scale"])
    if variant == "group32":
        kw["q_groups"] = 4
    if variant == "segments":
        kw["segm"] = _segm(q.shape[0], 28)
    out = paged_attention(q, kq, vq, torch.empty_like(q), cu, sl, bt, 1, SCALE, st, **kw)
    err = compare(ref, out)
    record("triton_int8", variant=variant, **err)
    assert err["rel_l2"] < 5e-3


def test_segments_equal_single_pass_bf16():
    q, k, v, cu, sl, bt = random_paged_case(28, 4, *CASES["decode"])
    a = paged_attention(q, k, v, torch.empty_like(q), cu, sl, bt, 1, SCALE)
    b = paged_attention(q, k, v, torch.empty_like(q), cu, sl, bt, 1, SCALE, segm=_segm(q.shape[0], 28))
    assert compare(a, b)["rel_l2"] < 2e-3


@pytest.fixture(params=DUMPS, ids=[n for n, _ in DUMPS])
def dump(request):
    return request.param


def test_int8_kernel_on_real_dumps(dump):
    """Real Qwen2.5 decode batches: kernel vs PyTorch implementation B vs captured FA2 output."""
    name, d = dump
    st = make_static(d["k_cache"], d["v_cache"], block_table=d["block_table"], seq_lens=d["seq_lens"])
    kq, vq = to_int8(d["k_cache"], d["v_cache"], st)
    args = (d["cu_seqlens_q"], d["seq_lens"], d["block_table"])
    sc = d["softmax_scale"].item()
    res = {}
    for label, cfg_name, kw in (("token_head", "B_channel_smooth", {}),
                                ("group32", "B_channel_smooth_qgroup32", {"q_groups": 4})):
        ref = paged_static_attention(CONFIGS[cfg_name], st, cache_is_int8=True)(d["q"], kq, vq, *args, sc)
        out = paged_attention(d["q"], kq, vq, torch.empty_like(d["q"]), *args, 1, sc, st, **kw)
        res[f"{label}_vs_ref"] = compare(ref, out)["rel_l2"]
        res[f"{label}_vs_fa2"] = compare(d["output"], out)["rel_l2"]
        res[f"{label}_ref_vs_fa2"] = compare(d["output"], ref)["rel_l2"]
    record("triton_int8_dump", case=name, **res)
    assert res["token_head_vs_ref"] < 5e-3 and res["group32_vs_ref"] < 5e-3


def test_int8_mode_for_verify_batches():
    """Spec-decode verification (k+1 query tokens per request) through the INT8 path."""
    q, k, v, cu, sl, bt = random_paged_case(28, 4, [5] * 6, [40, 700, 1500, 5, 333, 2048], scale=3.0)
    st = make_static(k, v)
    kq, vq = to_int8(k, v, st)
    cfg = CONFIGS["B_channel_smooth"]
    ks, vs = st["k_scale"][None, None], st["v_scale"][None, None]
    from kvq.int8_ref import attend_int8
    att = lambda qb, kk, vv, mask, valid: attend_int8(qb, kk, ks, vv, vs, mask, SCALE, cfg)  # noqa: E731
    ref = run_paged(att, q, kq, vq, cu, sl, bt, attend_multi=att)
    out = paged_attention(q, kq, vq, torch.empty_like(q), cu, sl, bt, 5, SCALE, st, int8_max_q=8)
    err = compare(ref, out)
    record("triton_int8_verify", **err)
    assert err["rel_l2"] < 5e-3
