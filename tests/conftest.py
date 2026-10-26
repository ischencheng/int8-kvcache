import json
import os
from collections import defaultdict

import pytest
import torch
from safetensors.torch import load_file

from kvq import quant
from kvq.ref_attn import gather_pages

TAGS = ("qwen2.5-7b", "qwen2.5-1.5b")


def record(test, **metrics):
    """Append metrics to $OUT/test_metrics.jsonl so runs leave numbers, not just PASS."""
    out = os.environ.get("OUT")
    if out:
        with open(os.path.join(out, "test_metrics.jsonl"), "a") as f:
            f.write(json.dumps({"test": test, **metrics}) + "\n")


def load_dumps():
    dumps = []
    for tag in TAGS:
        path = f"/cache/kv_capture/{tag}/decode_dumps.safetensors"
        if not os.path.exists(path):
            continue
        by_layer = defaultdict(dict)
        for key, t in load_file(path, device="cuda").items():
            layer, field = key.split("|")
            by_layer[layer][field] = t
        dumps += [(f"{tag}:{layer.split('.')[2]}", d) for layer, d in sorted(by_layer.items())]
    return dumps


DUMPS = load_dumps()


@pytest.fixture(params=DUMPS, ids=[name for name, _ in DUMPS])
def dump(request):
    return request.param


def random_paged_case(num_heads, num_kv_heads, q_lens, seq_lens, head_dim=128, block_size=16, seed=0,
                      dtype=torch.bfloat16, scale=1.0):
    """Random q and a paged KV cache whose pages are scattered in a larger pool."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    q_lens_t = torch.tensor(q_lens, dtype=torch.int32, device=dev)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    cu = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.cumsum(q_lens_t, 0)
    pages = [(s + block_size - 1) // block_size for s in seq_lens]
    num_blocks = 2 * sum(pages) + 8
    perm = torch.randperm(num_blocks, generator=g, device=dev).to(torch.int32)
    bt = torch.zeros(len(seq_lens), max(pages), dtype=torch.int32, device=dev)
    off = 0
    for i, p in enumerate(pages):
        bt[i, :p] = perm[off:off + p]
        off += p
    q = (torch.randn(int(cu[-1]), num_heads, head_dim, generator=g, device=dev) * scale).to(dtype)
    k = (torch.randn(num_blocks, block_size, num_kv_heads, head_dim, generator=g, device=dev) * scale).to(dtype)
    v = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, generator=g, device=dev).to(dtype)
    return q, k, v, cu, seq_lens_t, bt


# (q_lens, seq_lens) of the random paged batches
CASES = {
    "decode": ([1] * 12, [17, 300, 1, 1024, 2047, 64, 129, 900, 3000, 5, 333, 1500]),
    "short": ([2, 4, 8, 3, 1, 5, 7, 1, 6], [40, 700, 1029, 8, 2047, 96, 3000, 15, 512]),
    "mixed": ([1, 1, 7, 33, 128, 1, 300, 9, 4], [600, 31, 7, 500, 128, 1500, 900, 2000, 64]),
    "prefill": ([256, 97], [256, 1200]),
}


def _serving_case(seed=0):
    """Like a real chunked-prefill step: 68 prompt chunks among 188 decodes, interleaved (the batch whose
    FlashAttention call wrote out of bounds while short requests had zero keys)."""
    g = torch.Generator().manual_seed(seed)
    q = [int(x) for x in torch.randint(39, 74, (68,), generator=g)] + [1] * 188
    s = [n + int(torch.randint(0, 20, (1,), generator=g)) for n in q[:68]]
    s += [int(x) for x in torch.randint(100, 400, (188,), generator=g)]
    perm = torch.randperm(256, generator=g).tolist()
    return [q[i] for i in perm], [s[i] for i in perm]


CASES["serving"] = _serving_case()


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


def segm_buffers(T, hq, S=16, D=128):
    """Split-KV scratch like vLLM's Triton metadata builder allocates: [tokens, q heads, segments(, D)]."""
    return (torch.empty(T, hq, S, D, device="cuda"), torch.empty(T, hq, S, device="cuda"),
            torch.empty(T, hq, S, device="cuda"))
