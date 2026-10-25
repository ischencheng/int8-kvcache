"""Accuracy run: one model configuration on one task.

    python3 scripts/run_eval.py --task gsm8k --n 250 --label fa2
    python3 scripts/run_eval.py --task gsm8k --n 250 --label ref --env KVQ_ATTN=ref

--env entries are exported before vLLM starts, so the kvq plugin sees them in
every worker process. Writes $OUT/eval_<label>_<task>.json (+ .jsonl outputs).
"""
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k", "aime24"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--samples", type=int, default=1, help="samples per prompt (temperature > 0)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--label", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--llm", default="{}", help="extra LLM(...) kwargs as JSON")
    args = ap.parse_args()
    for kv in args.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v

    from vllm import LLM, SamplingParams

    from kvq import data, evaluate

    rows = data.gsm8k(args.n) if args.task == "gsm8k" else data.aime24()
    kwargs = dict(max_model_len=args.max_model_len, gpu_memory_utilization=0.85, seed=0)
    kwargs.update(json.loads(args.llm))
    llm = LLM(args.model, **kwargs)
    sp = SamplingParams(temperature=args.temperature, top_p=0.95 if args.temperature > 0 else 1.0,
                        max_tokens=args.max_tokens, n=args.samples, seed=0)
    t0 = time.time()
    outs = llm.chat([data.math_messages(r["question"]) for r in rows], sp)
    wall = time.time() - t0
    items = []
    for r, o in zip(rows, outs):
        for c in o.outputs:
            items.append({"id": r["id"], "answer": r["answer"], "correct": evaluate.math_correct(c.text, r["answer"]),
                          "tokens": len(c.token_ids), "text": c.text})
    acc = sum(i["correct"] for i in items) / len(items)
    res = {"label": args.label, "task": args.task, "model": args.model, "n_prompts": len(rows),
           "n_samples": len(items), "accuracy": acc, "wall_s": wall,
           "gen_tokens": sum(i["tokens"] for i in items), "env": args.env, "llm": kwargs}
    print(json.dumps(res, indent=1))
    out = os.environ["OUT"]
    with open(f"{out}/eval_{args.label}_{args.task}.json", "w") as f:
        json.dump(res, f, indent=1)
    with open(f"{out}/eval_{args.label}_{args.task}.jsonl", "w") as f:
        for i in items:
            f.write(json.dumps(i) + "\n")


if __name__ == "__main__":
    main()
