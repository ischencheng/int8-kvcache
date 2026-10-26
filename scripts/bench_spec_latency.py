"""Batch-1 decode latency with real prompts (spec-decode speedup is a low-batch effect).

Runs N GSM8K questions one at a time and reports the mean per-request TPOT from the engine's
first/last token timestamps, plus spec-decode acceptance.
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--llm", default="{}")
    args = ap.parse_args()
    for kv in args.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    from vllm import LLM, SamplingParams

    from kvq import data

    kwargs = dict(max_model_len=4096, gpu_memory_utilization=0.85, seed=0, disable_log_stats=False)
    kwargs.update(json.loads(args.llm))
    llm = LLM(args.model, **kwargs)
    sp = SamplingParams(temperature=0, max_tokens=512)
    llm.chat([data.math_messages("1+1=?")], sp, use_tqdm=False)  # warm up
    tpots = []
    for r in data.gsm8k(args.n, seed=7):
        o = llm.chat([data.math_messages(r["question"])], sp, use_tqdm=False)[0]
        m = o.metrics
        if m is not None and m.num_generation_tokens > 1:
            tpots.append((m.last_token_ts - m.first_token_ts) / (m.num_generation_tokens - 1))
    res = {"label": args.label, "model": args.model, "n": len(tpots), "tpot_ms": 1e3 * sum(tpots) / len(tpots)}
    print(json.dumps(res))
    with open(f"{os.environ['OUT']}/latency_{args.label}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
