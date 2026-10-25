"""Stage 2: look at the captured K/V and pick the quantization granularity.

For every layer: where the large values live (channel / head / token spread),
and the INT8 round-trip error ||x - Q(x)|| / ||x|| at each granularity.
Plots and a markdown table go to $OUT.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from kvq import quant
from kvq.capture import layer_index

OUT = os.environ["OUT"]


def spread(amax_hd):
    """Max/median channel ratio inside a head (worst head) and max/min across heads."""
    ch = (amax_hd.amax(1, keepdim=True) / amax_hd.median(1, keepdim=True).values).amax().item()
    head = amax_hd.amax(1)
    return ch, (head.max() / head.min()).item()


def layer_report(k, v, st):
    r = {}
    for side, x, amax in (("k", k, st["k_absmax"]), ("v", v, st["v_absmax"])):
        ch, hd = spread(amax)
        tok = x.abs().amax(dim=(1, 2))
        r[side] = {
            "absmax": amax.max().item(),
            "channel_ratio": ch,
            "head_ratio": hd,
            "token_absmax_p50": tok.median().item(),
            "token_absmax_max": tok.max().item(),
            "err": {g: quant.rel_error(x, quant.quant_dequant(x, g)[0]) for g in quant.GRANULARITIES},
            "err_per_channel_q999": quant.rel_error(x, quant.quant_dequant(x, "per_channel", 0.999)[0]),
        }
    # Smooth-K: a per-channel offset shared by every key adds the same q.mu to all
    # scores of a query, so softmax ignores it. Quantize k - mu, and measure the
    # error on the part attention can see (k minus its token mean).
    mu = k.mean(0, keepdim=True)
    ks = k - mu
    r["k"]["err_visible_per_channel"] = quant.rel_error(ks, quant.quant_dequant(k, "per_channel")[0] - mu)
    r["k"]["err_visible_per_channel_smooth"] = quant.rel_error(ks, quant.quant_dequant(ks, "per_channel")[0])
    r["k"]["err_visible_per_token_head"] = quant.rel_error(ks, quant.quant_dequant(k, "per_token_head")[0] - mu)
    # Qwen2 uses neox RoPE: channel i rotates with channel i + D/2, so post-RoPE
    # the pair shares its magnitude budget.
    a = st["k_absmax"]
    d2 = a.shape[1] // 2
    r["k_rope_pair_corr"] = torch.corrcoef(torch.stack([a[:, :d2].flatten(), a[:, d2:].flatten()]))[0, 1].item()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen2.5-7b")
    args = ap.parse_args()
    cap = torch.load(f"/cache/kv_capture/{args.tag}/capture.pt")
    names = sorted(cap["stats"], key=layer_index)
    reports = {}
    for name in names:
        k = cap["samples"][name]["k"].float()
        v = cap["samples"][name]["v"].float()
        reports[layer_index(name)] = layer_report(k, v, cap["stats"][name])
    L = sorted(reports)

    # Error per granularity across layers.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, side in zip(axes, "kv"):
        for g in quant.GRANULARITIES:
            ax.semilogy(L, [reports[i][side]["err"][g] for i in L], marker=".", label=g)
        ax.semilogy(L, [reports[i][side]["err_per_channel_q999"] for i in L], "--", label="per_channel p99.9")
        ax.set_title(f"{side.upper()} INT8 round-trip rel. error ({args.tag})")
        ax.set_xlabel("layer")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/err_by_granularity_{args.tag}.png", dpi=120)

    # Where the big values are: |K| / |V| channel max heatmaps.
    pick = [L[0], L[len(L) // 2], L[-1]]
    fig, axes = plt.subplots(len(pick), 2, figsize=(14, 2.2 * len(pick) + 1))
    for row, i in enumerate(pick):
        st = cap["stats"][names[i]]
        for col, side in enumerate("kv"):
            im = axes[row, col].imshow(st[f"{side}_absmax"].numpy(), aspect="auto", cmap="magma")
            axes[row, col].set_title(f"layer {i} |{side.upper()}| max per (head, channel)")
            axes[row, col].set_ylabel("kv head")
            fig.colorbar(im, ax=axes[row, col])
    axes[-1, 0].set_xlabel("channel")
    axes[-1, 1].set_xlabel("channel")
    fig.tight_layout()
    fig.savefig(f"{OUT}/absmax_heatmap_{args.tag}.png", dpi=120)

    # Per-channel distribution in the layer with the worst K channel spread.
    worst = max(L, key=lambda i: reports[i]["k"]["channel_ratio"])
    k = cap["samples"][names[worst]]["k"].float()
    h = int(cap["stats"][names[worst]]["k_absmax"].amax(1).argmax())
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.boxplot(k[:, h, :].numpy(), showfliers=False, whis=(0.1, 99.9))
    ax.set_xticks(range(0, k.shape[2] + 1, 16), range(0, k.shape[2] + 1, 16))
    ax.set_title(f"layer {worst}, kv head {h}: K value per channel over tokens (box 25-75%, whiskers 0.1-99.9%)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/k_channel_box_{args.tag}.png", dpi=120)

    rows = ["| layer | K max | K ch ratio | K head ratio | V max | V ch ratio | "
            + " | ".join(f"K {g}" for g in quant.GRANULARITIES) + " | "
            + " | ".join(f"V {g}" for g in quant.GRANULARITIES) + " |",
            "|" + "---|" * (6 + 2 * len(quant.GRANULARITIES))]
    for i in L:
        r = reports[i]
        rows.append(f"| {i} | {r['k']['absmax']:.1f} | {r['k']['channel_ratio']:.1f} | {r['k']['head_ratio']:.1f} | "
                    f"{r['v']['absmax']:.2f} | {r['v']['channel_ratio']:.1f} | "
                    + " | ".join(f"{r['k']['err'][g]:.4f}" for g in quant.GRANULARITIES) + " | "
                    + " | ".join(f"{r['v']['err'][g]:.4f}" for g in quant.GRANULARITIES) + " |")
    mean = {side: {g: sum(reports[i][side]["err"][g] for i in L) / len(L) for g in quant.GRANULARITIES}
            for side in "kv"}
    for key in ("err_visible_per_channel", "err_visible_per_channel_smooth", "err_visible_per_token_head"):
        mean["k"][key] = sum(reports[i]["k"][key] for i in L) / len(L)
    with open(f"{OUT}/kv_analysis_{args.tag}.md", "w") as f:
        f.write("\n".join(rows) + "\n\nmean over layers: " + json.dumps(mean) + "\n")
    with open(f"{OUT}/kv_analysis_{args.tag}.json", "w") as f:
        json.dump({"layers": reports, "mean_err": mean}, f, indent=1)
    print("\n".join(rows))
    print("mean over layers:", json.dumps(mean, indent=1))


if __name__ == "__main__":
    main()
