"""Collect results/*/ JSON into markdown tables and plots (run locally, no GPU).

    python3 scripts/report.py accuracy results/05-dynamic-* > docs/tables/stage5.md
    python3 scripts/report.py kernels results/08-bench-a100/bench_kernels_*.json
    python3 scripts/report.py e2e results/09-e2e-*
"""
import glob
import json
import math
import os
import sys


def _load(path):
    with open(path) as f:
        return json.load(f)


def accuracy(dirs):
    """GSM8K accuracy, flips vs bf16 and teacher-forced NLL deltas for every eval in the dirs."""
    rows, base_nll = {}, None
    for d in dirs:
        for p in glob.glob(f"{d}/eval_*_gsm8k.json"):
            r = _load(p)
            rows.setdefault(r["label"], {})["acc"] = r["accuracy"]
            rows[r["label"]]["n"] = r["n_samples"]
        for p in glob.glob(f"{d}/compare_fa2_*.json"):
            c = _load(p)
            label = os.path.basename(p)[len("compare_fa2_"):-5]
            rows.setdefault(label, {}).update(gained=c["gained"], lost=c["lost"], same=c["identical_outputs"])
        for p in glob.glob(f"{d}/nll_*.json"):
            r = _load(p)
            rows.setdefault(r["label"], {})["nll"] = r["nll"]
            if r["label"] == "fa2":
                base_nll = r["nll"]
    lines = ["| config | GSM8K acc | +/- vs bf16 (flips) | identical outputs | ΔNLL gsm8k | ΔNLL aime24 |",
             "|---|---|---|---|---|---|"]
    for label, r in sorted(rows.items(), key=lambda kv: (kv[0] != "fa2", kv[0])):
        acc = f"{100 * r['acc']:.2f}%" if "acc" in r else "-"
        flips = f"+{r['gained']} / -{r['lost']}" if "gained" in r else "-"
        same = f"{100 * r['same']:.1f}%" if "same" in r else "-"
        dn = ["-", "-"]
        if base_nll and "nll" in r:
            dn = [f"{r['nll'][k] - base_nll[k]:+.4f}" for k in ("gsm8k", "aime24")]
        lines.append(f"| {label} | {acc} | {flips} | {same} | {dn[0]} | {dn[1]} |")
    print("\n".join(lines))


def kernels(paths):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # tables only (plots are made in the container)
        plt = None

    for path in paths:
        res = _load(path)
        rows = res["decode"]
        names = [k for k in rows[0] if k not in ("batch", "ctx") and not k.startswith("kv_bytes")
                 and not k.endswith("_error")]
        print(f"\n### {res['gpu']}: decode attention, us (lower is better)\n")
        # speedup columns whose two kernels are in this file (our Triton kernel only before tag triton-backend)
        ratios = [(lbl, a, b) for lbl, a, b in (("triton int8 / triton bf16", "ours_bf16", "ours_int8"),
                                                ("cuda int8 vs fa2", "fa2_bf16", "cuda_int8"),
                                                ("cuda int8 vs fa3 fp8", "fa3_fp8", "cuda_int8"))
                  if a in names and b in names]
        extra = [lbl for lbl, _, _ in ratios]
        print("| batch | ctx | " + " | ".join(names) + " | " + " | ".join(extra) + " |")
        print("|---|---|" + "---|" * (len(names) + len(extra)))
        for r in rows:
            cells = [f"{r[n]:.1f}" if r.get(n) else "err" for n in names]
            sp = [r[a] / r[b] if r.get(a) and r.get(b) else math.nan for _, a, b in ratios]
            print(f"| {r['batch']} | {r['ctx']} | " + " | ".join(cells) + " | "
                  + " | ".join(f"{x:.2f}x" for x in sp) + " |")
        print("\nachieved KV bandwidth, GB/s (bytes the kernel must read / time)\n")
        # bytes each kernel must read: bf16 or int8 pages (int8_per_token_head adds its inline scales)
        nbytes = {"fa2_bf16": "kv_bytes_bf16", "ours_bf16": "kv_bytes_bf16", "ours_int8": "kv_bytes_int8",
                  "cuda_int8": "kv_bytes_int8", "vllm_int8_pth": "kv_bytes_int8"}
        bw_names = [n for n in nbytes if n in names]
        print("| batch | ctx | " + " | ".join(bw_names) + " |")
        print("|---|---|" + "---|" * len(bw_names))
        for r in rows:
            def bw(name):
                b = r[nbytes[name]] * (1.03 if name == "vllm_int8_pth" else 1)
                return f"{b / (r[name] * 1e-6) / 1e9:.0f}" if r.get(name) else "-"
            print(f"| {r['batch']} | {r['ctx']} | " + " | ".join(bw(n) for n in bw_names) + " |")
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2)) if plt else (None, [])
        for ax, b in zip(axes, (1, 32, 128)):
            sub = [r for r in rows if r["batch"] == b]
            for n in names:
                if all(r.get(n) for r in sub):
                    ax.plot([r["ctx"] for r in sub], [r[n] for r in sub], marker="o", label=n)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(f"{res['gpu'].split()[-1]} decode attention, batch {b}")
            ax.set_xlabel("context tokens")
            ax.set_ylabel("us")
            ax.grid(alpha=0.3)
        if plt:
            axes[0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(path.replace(".json", ".png"), dpi=120)
        if res.get("write"):
            print("\n### KV write path, us\n")
            keys = [k for k in res["write"][0] if k != "tokens"]
            print("| tokens | " + " | ".join(keys) + " |")
            print("|---|" + "---|" * len(keys))
            for r in res["write"]:
                print(f"| {r['tokens']} | " + " | ".join(f"{r[k]:.1f}" for k in keys) + " |")
        if res.get("q_quant"):
            print("\n### q quantization: standalone kernel vs fused into attention, us\n")
            keys = [k for k in res["q_quant"][0] if k not in ("batch", "ctx")]
            print("| batch | ctx | " + " | ".join(keys) + " |")
            print("|---|---|" + "---|" * len(keys))
            for r in res["q_quant"]:
                print(f"| {r['batch']} | {r['ctx']} | " + " | ".join(f"{r[k]:.1f}" for k in keys) + " |")


def e2e(dirs):
    runs = {}
    for d in dirs:
        for p in glob.glob(f"{d}/e2e_*.json"):
            r = _load(p)
            runs[os.path.basename(p)[4:-5]] = r
    labels = sorted(k for k in runs if not k.endswith("_tput"))
    if labels:
        pts = sorted({(r["ctx"], r["batch"]) for k in labels for r in runs[k]["tpot"]})
        print("| ctx | batch | " + " | ".join(labels) + " |")
        print("|---|---|" + "---|" * len(labels))
        for ctx, b in pts:
            cells = []
            for k in labels:
                m = [r for r in runs[k]["tpot"] if r["ctx"] == ctx and r["batch"] == b]
                key = next((k for k in ("tpot_pure_ms", "tpot_req_ms", "tpot_ms") if m and m[0].get(k)), None)
                cells.append(f"{m[0][key]:.2f}{'*' if m[0]['kv_limited'] else ''}" if m else "-")
            print(f"| {ctx} | {b} | " + " | ".join(cells) + " |")
        print("\nTPOT in ms (pure decode (T(2L) - T(L)) / L when recorded, else per-request engine timestamps, "
              "else (total - prefill) / steps); * = KV-limited (requests queue)\n")
        print("| config | KV capacity (tokens) |")
        print("|---|---|")
        for k in labels:
            print(f"| {k} | {runs[k]['kv_capacity_tokens']:,} |")
    tp = [k for k in runs if k.endswith("_tput")]
    if tp:
        print("\n| config | requests x (ctx + out) | max concurrent by KV | output tok/s |")
        print("|---|---|---|---|")
        for k in sorted(tp):
            t = runs[k]["throughput"]
            print(f"| {k[:-5]} | {t['requests']} x ({t['ctx']} + {t['out']}) | {t['max_concurrent_by_kv']} | "
                  f"{t['output_tok_per_s']:.0f} |")


if __name__ == "__main__":
    {"accuracy": accuracy, "kernels": kernels, "e2e": e2e}[sys.argv[1]](sys.argv[2:])
