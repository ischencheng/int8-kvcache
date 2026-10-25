"""MLA: latent distribution, per-granularity INT8 error, and decode-attention error on real dumps."""
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file

from kvq import mla, quant
from kvq.ref_attn import compare

ROOT = "/cache/mla/dsv2-lite"
OUT = os.environ["OUT"]


def main():
    cap = torch.load(f"{ROOT}/capture.pt")
    layers = sorted(cap["stats"])
    rep = {}
    for i in layers:
        r = {}
        for side in ("c", "kpe"):
            x = cap["samples"][i][side].float()[:, None, :]           # [T, 1, C] as one "head"
            amax = cap["stats"][i][side]["absmax"]
            mu = x.mean(0, keepdim=True)
            r[side] = {"absmax": amax.max().item(),
                       "channel_ratio": (amax.max() / amax.median()).item(),
                       "err": {g: quant.rel_error(x, quant.quant_dequant(x, g)[0])
                               for g in ("per_tensor", "per_channel", "per_token")},
                       "err_per_channel_smooth": quant.rel_error(x, quant.quant_dequant(x - mu, "per_channel")[0] + mu)}
        rep[i] = r
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, side in zip(axes, ("c", "kpe")):
        for g in ("per_tensor", "per_channel", "per_token"):
            ax.semilogy(layers, [rep[i][side]["err"][g] for i in layers], marker=".", label=g)
        ax.semilogy(layers, [rep[i][side]["err_per_channel_smooth"] for i in layers], "--", label="per_channel smooth")
        ax.set_title(f"DeepSeek-V2-Lite {'latent c (512)' if side == 'c' else 'k_pe (64, RoPE)'}: INT8 rel. error")
        ax.set_xlabel("layer")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/mla_err_by_granularity.png", dpi=120)

    # Decode attention on the dumps, against the captured Triton MLA output.
    st_plain = mla.load_static(f"{ROOT}/scales.safetensors")
    st_smooth = mla.load_static(f"{ROOT}/scales_smooth.safetensors")
    dumps = defaultdict(dict)
    for k, t in load_file(f"{ROOT}/decode_dumps.safetensors", device="cuda").items():
        i, f = k.split("|")
        dumps[int(i)][f] = t
    cfgs = {
        "A_plain": ({"impl": "A", "smooth": False, "rope_int8": False}, st_plain),
        "A_smooth": ({"impl": "A", "smooth": True, "rope_int8": False}, st_smooth),
        "A_smooth_rope8": ({"impl": "A", "smooth": True, "rope_int8": True}, st_smooth),
        "B_plain": ({"impl": "B", "smooth": False, "rope_int8": False}, st_plain),
        "B_smooth": ({"impl": "B", "smooth": True, "rope_int8": False}, st_smooth),
        "B_smooth_rope8": ({"impl": "B", "smooth": True, "rope_int8": True}, st_smooth),
    }
    op = {}
    for i, d in sorted(dumps.items()):
        args = (d["q_lat"], d["q_pe"], d["cache"], d["block_table"], d["seq_lens"], d["scale"].item())
        row = {"ref_vs_kernel": compare(d["o"], mla.decode_ref(*args))["rel_l2"]}
        for name, (cfg, st) in cfgs.items():
            row[name] = compare(d["o"], mla.decode_ref(*args, cfg=cfg, static=st[i]))["rel_l2"]
        op[i] = row
        print(i, json.dumps({k: round(v, 5) for k, v in row.items()}))
    with open(f"{OUT}/mla_analysis.json", "w") as f:
        json.dump({"layers": rep, "decode_op_error": op}, f, indent=1)


if __name__ == "__main__":
    main()
