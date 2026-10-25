"""Extension 3.5(b): the INT8 kernels under CUDA graph capture.

A captured graph replays the same kernels with the same pointers. So (1) every input
that changes per step must be copied into the captured buffers, and (2) the static
scales must stay the same tensors for the life of the graph: rebinding them to new
tensors silently keeps the old values, only an in-place copy_ reaches the graph.
"""
import torch
from conftest import random_paged_case, record
from test_triton_kernels import make_static, to_int8

from kvq.ref_attn import compare
from kvq.triton_kernels.int8_attention import paged_attention
from kvq.triton_kernels.quant_cache import quant_reshape_and_cache

SCALE = 128 ** -0.5
Q_LENS = [1] * 8
SEQ_LENS = [100, 2000, 37, 512, 1500, 64, 999, 3000]


def _capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()  # warm up (and JIT-compile) outside the capture
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def test_replay_matches_eager_with_new_inputs():
    q, k, v, cu, sl, bt = random_paged_case(28, 4, Q_LENS, SEQ_LENS, scale=3.0)
    st = make_static(k, v)
    kq, vq = to_int8(k, v, st)
    out = torch.empty_like(q)
    g = _capture(lambda: paged_attention(q, kq, vq, out, cu, sl, bt, 1, SCALE, st))
    # Next "step": new queries, one more token per request (same shapes, new contents).
    q2, *_ = random_paged_case(28, 4, Q_LENS, SEQ_LENS, scale=3.0, seed=7)
    q.copy_(q2)
    sl.add_(1)
    g.replay()
    ref = paged_attention(q, kq, vq, torch.empty_like(q), cu, sl, bt, 1, SCALE, st)
    err = compare(ref, out)
    record("cudagraph_replay", **err)
    assert err["max_abs"] == 0.0


def test_scales_must_be_updated_in_place():
    q, k, v, cu, sl, bt = random_paged_case(28, 4, Q_LENS, SEQ_LENS, scale=3.0)
    st = make_static(k, v)
    kq, vq = to_int8(k, v, st)
    out = torch.empty_like(q)
    g = _capture(lambda: paged_attention(q, kq, vq, out, cu, sl, bt, 1, SCALE, st))
    g.replay()
    before = out.clone()
    new_v_scale = st["v_scale"] * 2          # e.g. reloading a new calibration
    st["v_scale"] = new_v_scale              # rebinding: the graph still reads the old tensor
    g.replay()
    stale = torch.equal(out, before)
    expected = paged_attention(q, kq, vq, torch.empty_like(q), cu, sl, bt, 1, SCALE, st)
    record("cudagraph_rebind_scale", stale_after_rebind=stale)
    assert stale and not torch.allclose(out, expected)


def test_scales_copy_in_place_reaches_graph():
    q, k, v, cu, sl, bt = random_paged_case(28, 4, Q_LENS, SEQ_LENS, scale=3.0)
    st = make_static(k, v)
    kq, vq = to_int8(k, v, st)
    out = torch.empty_like(q)
    g = _capture(lambda: paged_attention(q, kq, vq, out, cu, sl, bt, 1, SCALE, st))
    st["v_scale"].mul_(2)                    # in place: same storage the graph captured
    g.replay()
    expected = paged_attention(q, kq, vq, torch.empty_like(q), cu, sl, bt, 1, SCALE, st)
    assert compare(expected, out)["max_abs"] == 0.0


def test_quant_cache_capture_skips_padded_slots():
    """Under a full graph the slot mapping is padded with -1 up to the capture size."""
    T, H, D, nb, bs = 64, 4, 128, 32, 16
    key = torch.randn(T, H, D, device="cuda").to(torch.bfloat16)
    val = torch.randn(T, H, D, device="cuda").to(torch.bfloat16)
    slots = torch.full((T,), -1, dtype=torch.long, device="cuda")
    kc = torch.zeros(nb, bs, H, D, dtype=torch.int8, device="cuda")
    vc = torch.zeros_like(kc)
    st = make_static(key[:, None], val[:, None])
    g = _capture(lambda: quant_reshape_and_cache(key, val, kc, vc, slots, st))
    kc.zero_()
    vc.zero_()
    slots[:10] = torch.arange(10, device="cuda") * 3     # only 10 real tokens this step
    g.replay()
    assert int((kc != 0).flatten(2).any(-1).sum()) == 10
