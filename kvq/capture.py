"""Stage 2 instrumentation: record what vLLM writes into the KV cache.

Installed by the vLLM plugin when KVQ_CAPTURE=1. It wraps the FlashAttention
backend (the A100 default) and keeps, per attention layer:
  * running |K| / |V| max per (kv_head, channel), plus a sampled reservoir of
    K/V rows for distribution plots and quantile scales;
  * optionally a full dump of one decode call (q, used KV blocks, block table,
    lengths, output) for the stage-3 golden test (KVQ_DUMP_AT_SEQLEN).
Read back with collect() through LLM.collective_rpc.
"""
import os

import torch

STATE = {"stats": {}, "samples": {}, "dumps": {}}
MAX_SAMPLE_TOKENS = int(os.environ.get("KVQ_CAPTURE_MAX_TOKENS", "16384"))
DUMP_AT_SEQLEN = int(os.environ.get("KVQ_DUMP_AT_SEQLEN", "0"))
DUMP_LAYERS = {int(x) for x in os.environ.get("KVQ_DUMP_LAYERS", "").split(",") if x}


def layer_index(name):
    return int(name.split("layers.")[1].split(".")[0])


def _record(name, key, value):
    """Running per-(kv_head, channel) stats: |x| max, signed max/min and sum (for smooth-K)."""
    new = {}
    for side, x in (("k", key), ("v", value)):
        xf = x.float()
        new[f"{side}_absmax"] = xf.abs().amax(0)
        new[f"{side}_max"] = xf.amax(0)
        new[f"{side}_min"] = xf.amin(0)
        new[f"{side}_sum"] = xf.double().sum(0)
    st = STATE["stats"].get(name)
    if st is None:
        STATE["stats"][name] = st = {**new, "tokens": 0}
    else:
        for key_ in new:
            if key_.endswith("_sum"):
                st[key_] += new[key_]
            elif key_.endswith("_min"):
                torch.minimum(st[key_], new[key_], out=st[key_])
            else:
                torch.maximum(st[key_], new[key_], out=st[key_])
    st["tokens"] += key.shape[0]
    # Reservoir-free sampling: keep every token until the budget, then a random
    # 1/8 of each new batch. Enough for quantiles and plots.
    smp = STATE["samples"].setdefault(name, {"k": [], "v": [], "n": 0})
    if smp["n"] >= MAX_SAMPLE_TOKENS:
        return
    if smp["n"] > MAX_SAMPLE_TOKENS // 2:
        idx = torch.randperm(key.shape[0], device=key.device)[: max(1, key.shape[0] // 8)]
        key, value = key[idx], value[idx]
    smp["k"].append(key.to(torch.float16).cpu())
    smp["v"].append(value.to(torch.float16).cpu())
    smp["n"] += key.shape[0]


def _maybe_dump(impl, layer, query, kv_cache, md, output):
    name = layer.layer_name
    if not DUMP_AT_SEQLEN or name in STATE["dumps"] or layer_index(name) not in DUMP_LAYERS:
        return
    if md.max_query_len != 1 or md.max_seq_len < DUMP_AT_SEQLEN:
        return
    n = md.num_actual_tokens
    num_reqs = md.query_start_loc.shape[0] - 1
    seq_lens = md.seq_lens[:num_reqs]
    bt = md.block_table[:num_reqs]
    block_size = kv_cache.shape[2]
    pages = (seq_lens + block_size - 1) // block_size
    used = torch.cat([bt[i, : pages[i]] for i in range(num_reqs)])
    # Compact the used pages so the dump stays small; remap the block table.
    remap = torch.zeros(int(bt.max()) + 1, dtype=torch.int32, device=bt.device)
    remap[used] = torch.arange(used.numel(), dtype=torch.int32, device=bt.device)
    new_bt = torch.zeros_like(bt)
    for i in range(num_reqs):
        new_bt[i, : pages[i]] = remap[bt[i, : pages[i]]]
    kc, vc = kv_cache.transpose(1, 2).split(impl.head_size, dim=-1)  # (B, N, H, D)
    STATE["dumps"][name] = {
        "q": query[:n].cpu(), "k_cache": kc[used].contiguous().cpu(), "v_cache": vc[used].contiguous().cpu(),
        "block_table": new_bt.cpu(), "seq_lens": seq_lens.cpu(), "cu_seqlens_q": md.query_start_loc[: num_reqs + 1].cpu(),
        "output": output[:n].cpu(),
        "softmax_scale": torch.tensor(impl.scale), "max_seqlen_k": torch.tensor(md.max_seq_len),
    }


def install():
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    orig_update = FlashAttentionImpl.do_kv_cache_update
    orig_forward = FlashAttentionImpl.forward

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        n = slot_mapping.shape[0]
        valid = slot_mapping >= 0
        _record(layer.layer_name, key[:n][valid], value[:n][valid])
        return orig_update(self, layer, key, value, kv_cache, slot_mapping)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
        out = orig_forward(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
        if attn_metadata is not None:
            _maybe_dump(self, layer, query, kv_cache, attn_metadata, output)
        return out

    FlashAttentionImpl.do_kv_cache_update = do_kv_cache_update
    FlashAttentionImpl.forward = forward


def collect(worker=None):
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    out = {"stats": {}, "samples": {}, "dumps": STATE["dumps"],
           "tp_rank": get_tensor_model_parallel_rank(), "tp_size": get_tensor_model_parallel_world_size()}
    for name, st in STATE["stats"].items():
        out["stats"][name] = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in st.items()}
    for name, smp in STATE["samples"].items():
        out["samples"][name] = {"k": torch.cat(smp["k"]), "v": torch.cat(smp["v"])}
    return out
