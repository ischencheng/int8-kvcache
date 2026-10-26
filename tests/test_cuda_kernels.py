"""Stage 7 (CUDA): the CUDA kernels and the long-query path against the PyTorch references."""
import pytest
import torch
from conftest import CASES, DUMPS, make_static, random_paged_case, record, segm_buffers, to_int8

from kvq import cuda_kernels as ck
from kvq import quant
from kvq.int8_ref import Cfg, paged_static_attention
from kvq.ref_attn import compare

SCALE = 128 ** -0.5
CFG32 = Cfg(smooth_k=True, p_tile=32)            # the CUDA kernel quantizes P per 32-token chunk


def reference(st, per_channel=True):
    cfg = CFG32 if per_channel else Cfg(smooth_k=True, p_tile=32, k_gran="per_head", v_gran="per_head")
    return paged_static_attention(cfg, st, cache_is_int8=True, short_q=ck.SHORT_Q)


def short_rows(cu, limit=ck.SHORT_Q):
    """Token mask of the requests the INT8 kernel handles."""
    q_lens = (cu[1:] - cu[:-1]).long()
    return torch.repeat_interleave(q_lens <= limit, q_lens)


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
    ck.quant_reshape_and_cache(key, value, kc, vc, slots, st)
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
    qi, qs = ck.quant_q(q, ks, group)
    qf = (q.float().reshape(37, 4, 7, 128) * ks[None, :, None, :]).reshape(37, 28, 128 // group, group)
    s = quant.scale_from_absmax(qf.abs().amax(-1, keepdim=True))
    assert torch.equal(qi.reshape(37, 28, -1, group), quant.quantize(qf, s))
    assert torch.allclose(qs, s[..., 0])


@pytest.mark.parametrize("heads", [(28, 4), (12, 2), (32, 8), (16, 1)])
@pytest.mark.parametrize("case", ["decode", "short"])
@pytest.mark.parametrize("variant", ["single", "segments", "per_head"])
def test_short_queries_match_ref(heads, case, variant):
    """Decode and short multi-token queries (causal inside the request, several m tiles when q_len * G > 16)."""
    q_lens, seq_lens = CASES[case]
    q, k, v, cu, sl, bt = random_paged_case(*heads, q_lens, seq_lens, scale=3.0)
    st = make_static(k, v, per_channel=variant != "per_head")
    kq, vq = to_int8(k, v, st)
    ref = reference(st, variant != "per_head")(q, kq, vq, cu, sl, bt, SCALE)
    out = ck.decode_attention(q, kq, vq, torch.empty_like(q), cu, sl, bt, SCALE, st,
                              segm=segm_buffers(q.shape[0], heads[0]) if variant == "segments" else None,
                              max_query_len=max(q_lens))
    err = compare(ref, out)
    record("cuda_short", heads=heads, case=case, variant=variant, **err)
    assert err["rel_l2"] < 5e-3


def test_short_kernel_leaves_long_requests_alone():
    q, k, v, cu, sl, bt = random_paged_case(28, 4, *CASES["mixed"], scale=3.0)
    st = make_static(k, v)
    kq, vq = to_int8(k, v, st)
    out = torch.full_like(q, 7.0)
    ck.decode_attention(q, kq, vq, out, cu, sl, bt, SCALE, st, max_query_len=int((cu[1:] - cu[:-1]).max()))
    short = short_rows(cu)
    assert (out[~short] == 7.0).all() and not (out[short] == 7.0).any()


def test_plan_long_page_table():
    cu = torch.tensor([0, 1, 13, 14, 314], dtype=torch.int32)        # q_lens 1, 12, 1, 300
    sl = torch.tensor([50, 40, 9, 330], dtype=torch.int32)
    bt = torch.arange(4 * 21, dtype=torch.int32, device="cuda").reshape(4, 21) + 1000
    plan = ck.plan_long(cu, sl, bt, 16)
    assert plan.seqused_k.tolist() == [0, 40, 0, 330] and plan.max_seqlen_k == 330 and plan.has_short
    assert plan.src_pages.tolist() == bt[1, :3].tolist() + bt[3, :21].tolist()
    assert plan.block_table[1, :3].tolist() == [0, 1, 2] and plan.block_table[3, :21].tolist() == list(range(3, 24))
    assert ck.plan_long(torch.tensor([0, 1, 9]), torch.tensor([5, 8]), bt, 16) is None   # q_len 8 is short


@pytest.mark.parametrize("heads", [(28, 4), (12, 2)])
@pytest.mark.parametrize("case", ["mixed", "prefill"])
@pytest.mark.parametrize("per_channel", [True, False])
def test_mixed_batch_matches_ref(heads, case, per_channel):
    """Whole batch the way the backend runs it: long requests dequantize + FlashAttention, short ones INT8."""
    q_lens, seq_lens = CASES[case]
    q, k, v, cu, sl, bt = random_paged_case(*heads, q_lens, seq_lens, scale=3.0)
    st = make_static(k, v, per_channel=per_channel)
    kq, vq = to_int8(k, v, st)
    ref = reference(st, per_channel)(q, kq, vq, cu, sl, bt, SCALE)
    plan = ck.plan_long(cu.cpu(), sl.cpu(), bt, kq.shape[1])
    assert plan is not None and plan.has_short == (case == "mixed")
    out = ck.paged_attention(q, kq, vq, torch.empty_like(q), cu, sl, bt, max(q_lens), SCALE, st, plan)
    short = short_rows(cu)
    res = {"all": compare(ref, out)["rel_l2"], "long": compare(ref[~short], out[~short])["rel_l2"]}
    if short.any():
        res["short"] = compare(ref[short], out[short])["rel_l2"]
    record("cuda_mixed", heads=heads, case=case, per_channel=per_channel, **res)
    assert res["long"] < 1e-2 and res.get("short", 0) < 5e-3   # long rows: bf16 FlashAttention on bf16 K/V


def test_cuda_decode_on_real_dumps(dump):
    name, d = dump
    st = make_static(d["k_cache"], d["v_cache"], block_table=d["block_table"], seq_lens=d["seq_lens"])
    kq, vq = to_int8(d["k_cache"], d["v_cache"], st)
    args = (d["cu_seqlens_q"], d["seq_lens"], d["block_table"])
    sc = d["softmax_scale"].item()
    ref = paged_static_attention(CFG32, st, cache_is_int8=True)(d["q"], kq, vq, *args, sc)
    out = ck.decode_attention(d["q"], kq, vq, torch.empty_like(d["q"]), *args, sc, st)
    res = {"vs_ref": compare(ref, out)["rel_l2"], "vs_fa2": compare(d["output"], out)["rel_l2"]}
    record("cuda_decode_dump", case=name, **res)
    assert res["vs_ref"] < 5e-3
