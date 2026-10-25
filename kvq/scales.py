"""Static KV scales: compute from calibration stats, save/load, shard for TP.

File format (safetensors): "<layer_name>|k_scale", "|v_scale" and optional
"|k_mu", each [num_kv_heads_total, D] for per_channel or [num_kv_heads_total, 1]
for per_head. Scales are the dequantization step (x ~= q * scale).
"""
import torch
from safetensors.torch import load_file, save_file

from kvq import quant


def from_stats(st, samples=None, granularity="per_channel", method="max", smooth_k=False):
    """One layer's static scales from capture stats (+ sampled rows for quantiles)."""
    out = {}
    mu = None
    if smooth_k:
        mu = (st["k_sum"] / st["tokens"]).float()
        out["k_mu"] = mu
    for side in ("k", "v"):
        if side == "k" and mu is not None:
            amax = torch.maximum(st["k_max"] - mu, mu - st["k_min"])
        else:
            amax = st[f"{side}_absmax"].float()
        if method != "max":
            q = float(method[1:])                     # "q0.999"
            x = samples[side].float()
            if side == "k" and mu is not None:
                x = x - mu
            amax = quant.absmax(x, "per_channel", quantile=q)[0]
        if granularity == "per_head":
            amax = amax.amax(-1, keepdim=True)
        out[f"{side}_scale"] = quant.scale_from_absmax(amax)
    return out


def save(path, scales):
    save_file({f"{name}|{k}": v.contiguous().float() for name, d in scales.items() for k, v in d.items()}, path)


def load(path, device="cpu"):
    out = {}
    for key, t in load_file(path, device=device).items():
        name, field = key.split("|")
        out.setdefault(name, {})[field] = t
    return out


def shard(t, tp_rank, tp_size):
    """Rows (kv heads) of this TP rank. With fewer kv heads than ranks vLLM replicates them."""
    h = t.shape[0]
    if h >= tp_size:
        assert h % tp_size == 0
        n = h // tp_size
        return t[tp_rank * n:(tp_rank + 1) * n]
    return t[tp_rank * h // tp_size:tp_rank * h // tp_size + 1]


def merge_tp(per_rank, num_kv_heads):
    """Calibration under TP: each rank saw only its own kv heads, so stats and sampled rows
    are concatenated along heads in rank order. When kv heads < ranks, vLLM replicates each
    head on tp/num_kv_heads ranks; keep one replica."""
    ranks = sorted(per_rank, key=lambda r: r["tp_rank"])
    tp = len(ranks)
    if tp > num_kv_heads:
        ranks = ranks[:: tp // num_kv_heads]
    if len(ranks) == 1:
        return ranks[0]["stats"], ranks[0]["samples"]
    stats, samples = {}, {}
    for name in ranks[0]["stats"]:
        parts = [r["stats"][name] for r in ranks]
        stats[name] = {"tokens": parts[0]["tokens"],
                       **{k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0] if k != "tokens"}}
        samples[name] = {side: torch.cat([r["samples"][name][side] for r in ranks], dim=1) for side in "kv"}
    return stats, samples


def for_layer(all_scales, name, tp_rank, tp_size, device):
    """This rank's slice of one layer's scales, on device."""
    return {k: shard(v, tp_rank, tp_size).to(device) for k, v in all_scales[name].items()}
