"""Stage 3 levels 1 and 2: the PyTorch reference against vLLM's FA2 kernel."""
import pytest
import torch
from conftest import random_paged_case, record
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

from kvq.ref_attn import compare, paged_attention_ref


def run_fa(q, k, v, cu, seq_lens, bt, scale):
    out = torch.empty_like(q)
    flash_attn_varlen_func(q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=int((cu[1:] - cu[:-1]).max()),
                           seqused_k=seq_lens, max_seqlen_k=int(seq_lens.max()), softmax_scale=scale,
                           causal=True, block_table=bt, fa_version=2)
    return out


def test_vllm_reproduces_dump(dump):
    """The captured inputs must reproduce the captured output bit for bit,
    otherwise the dump (or how we call the kernel) is wrong."""
    name, d = dump
    out = run_fa(d["q"], d["k_cache"], d["v_cache"], d["cu_seqlens_q"], d["seq_lens"], d["block_table"],
                 d["softmax_scale"].item())
    record("vllm_reproduces_dump", case=name, batch=d["seq_lens"].numel(), max_seq=int(d["seq_lens"].max()),
           equal=bool(torch.equal(out, d["output"])))
    assert torch.equal(out, d["output"])


def test_ref_matches_dump(dump):
    name, d = dump
    ref = paged_attention_ref(d["q"], d["k_cache"], d["v_cache"], d["cu_seqlens_q"], d["seq_lens"],
                              d["block_table"], d["softmax_scale"].item())
    err = compare(ref, d["output"])
    record("ref_matches_dump", case=name, **err)
    assert err["rel_l2"] < 1e-2 and err["cos"] > 0.9999


@pytest.mark.parametrize("heads", [(28, 4), (12, 2)])
@pytest.mark.parametrize("q_lens,seq_lens", [
    ([1] * 8, [17, 300, 1, 1024, 2047, 64, 129, 900]),          # decode batch
    ([1, 1, 7, 33, 128, 1], [600, 31, 7, 500, 128, 1500]),       # mixed prefill chunks + decode
])
def test_ref_matches_fa_random(heads, q_lens, seq_lens):
    q, k, v, cu, sl, bt = random_paged_case(*heads, q_lens, seq_lens)
    scale = 128 ** -0.5
    err = compare(paged_attention_ref(q, k, v, cu, sl, bt, scale), run_fa(q, k, v, cu, sl, bt, scale))
    record("ref_matches_fa_random", heads=heads, q_lens=q_lens, **err)
    assert err["rel_l2"] < 5e-3 and err["cos"] > 0.99995
