"""Compare two eval runs item by item: accuracy delta, flips, identical outputs."""
import json
import sys


def load(path):
    return [json.loads(line) for line in open(path)]


def main(a_path, b_path):
    a, b = load(a_path), load(b_path)
    assert [x["id"] for x in a] == [x["id"] for x in b]
    same = sum(x["text"] == y["text"] for x, y in zip(a, b))
    gained = sum((not x["correct"]) and y["correct"] for x, y in zip(a, b))
    lost = sum(x["correct"] and not y["correct"] for x, y in zip(a, b))
    prefix = []
    for x, y in zip(a, b):
        n = 0
        for c1, c2 in zip(x["text"], y["text"]):
            if c1 != c2:
                break
            n += 1
        prefix.append(n / max(1, len(x["text"])))
    res = {"n": len(a), "acc_a": sum(x["correct"] for x in a) / len(a), "acc_b": sum(y["correct"] for y in b) / len(b),
           "identical_outputs": same / len(a), "gained": gained, "lost": lost,
           "mean_common_prefix_frac": sum(prefix) / len(prefix)}
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
