import json
import os
from collections import defaultdict

import pytest
import torch
from safetensors.torch import load_file

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
