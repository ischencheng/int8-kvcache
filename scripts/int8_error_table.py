"""Stage 4: attention-output error of every INT8 configuration on the real decode dumps.

Reference = the captured FA2 output. err(A) is pure KV quantization error;
err(B) - err(A) is what the INT8 arithmetic (q / P quantization) adds.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from conftest import DUMPS  # noqa: E402

from kvq.int8_ref import CONFIGS, paged_int8_attention  # noqa: E402
from kvq.ref_attn import compare, paged_attention_ref  # noqa: E402


def main():
    rows = []
    for case, d in DUMPS:
        args = (d["q"], d["k_cache"], d["v_cache"], d["cu_seqlens_q"], d["seq_lens"], d["block_table"],
                d["softmax_scale"].item())
        res = {"case": case, "ref_fp32": compare(d["output"], paged_attention_ref(*args))["rel_l2"]}
        for name, cfg in CONFIGS.items():
            res[name] = compare(d["output"], paged_int8_attention(cfg)(*args))["rel_l2"]
        rows.append(res)
        print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in res.items()}))
    names = ["ref_fp32"] + list(CONFIGS)
    lines = ["| case | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for r in rows:
        lines.append(f"| {r['case']} | " + " | ".join(f"{100 * r[n]:.2f}%" for n in names) + " |")
    mean = {n: sum(r[n] for r in rows) / len(rows) for n in names}
    lines.append("| **mean** | " + " | ".join(f"**{100 * mean[n]:.2f}%**" for n in names) + " |")
    out = os.environ["OUT"]
    with open(f"{out}/int8_error_table.md", "w") as f:
        f.write("attention output rel_l2 vs captured FA2 output\n\n" + "\n".join(lines) + "\n")
    with open(f"{out}/int8_error_table.json", "w") as f:
        json.dump({"rows": rows, "mean": mean}, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
