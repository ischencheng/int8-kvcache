"""Compare two static-scale files layer by layer (e.g. TP=1 vs TP=2 calibration)."""
import json
import os
import sys

from kvq import scales


def main(a_path, b_path):
    a, b = scales.load(a_path), scales.load(b_path)
    assert a.keys() == b.keys()
    worst = {}
    for name in a:
        for k in a[name]:
            x, y = a[name][k].float(), b[name][k].float()
            rel = ((x - y).abs() / x.abs().clamp(min=1e-6)).max().item()
            worst[k] = max(worst.get(k, 0.0), rel)
    res = {"a": a_path, "b": b_path, "layers": len(a), "max_rel_diff": worst}
    print(json.dumps(res, indent=1))
    with open(f"{os.environ['OUT']}/compare_scales.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
