"""Teacher-forced NLL of fixed reference answers under one model configuration.

Accuracy on a few hundred questions moves by whole questions; the mean NLL of
the bf16 model's own greedy answers moves smoothly with any distortion of the
attention, so it ranks KV-cache configurations with much less noise.

    python3 scripts/eval_nll.py --refs $OUT/eval_fa2_gsm8k.jsonl $OUT/eval_fa2_aime24.jsonl \
        --label A_channel --env KVQ_ATTN=A_channel
"""
import argparse
import json
import math
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--per-ref", type=int, default=200)
    ap.add_argument("--label", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--llm", default="{}")
    args = ap.parse_args()
    for kv in args.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from kvq import data

    questions = {r["id"]: r["question"] for r in data.gsm8k() + data.aime24()}
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts, spans, groups = [], [], []
    for path in args.refs:
        rows = [json.loads(line) for line in open(path)][: args.per_ref]
        for r in rows:
            head = tok.apply_chat_template(data.math_messages(questions[r["id"]]), tokenize=False,
                                           add_generation_prompt=True)
            p_ids = tok(head, add_special_tokens=False)["input_ids"]
            a_ids = tok(r["text"], add_special_tokens=False)["input_ids"]
            prompts.append(TokensPrompt(prompt_token_ids=p_ids + a_ids))
            spans.append((len(p_ids), len(p_ids) + len(a_ids)))
            groups.append(os.path.basename(path).split("_")[-1].split(".")[0])

    kwargs = dict(max_model_len=4096, gpu_memory_utilization=0.85, seed=0, enable_prefix_caching=False)
    kwargs.update(json.loads(args.llm))
    llm = LLM(args.model, **kwargs)
    outs = llm.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=0))
    per_group = {}
    for o, (s, e), grp, pr in zip(outs, spans, groups, prompts):
        ids = pr["prompt_token_ids"]
        nll = -sum(o.prompt_logprobs[i][ids[i]].logprob for i in range(s, e))
        g = per_group.setdefault(grp, [0.0, 0])
        g[0] += nll
        g[1] += e - s
    res = {"label": args.label, "env": args.env,
           "nll": {g: v[0] / v[1] for g, v in per_group.items()},
           "tokens": {g: v[1] for g, v in per_group.items()}}
    res["ppl"] = {g: math.exp(v) for g, v in res["nll"].items()}
    print(json.dumps(res, indent=1))
    with open(f"{os.environ['OUT']}/nll_{args.label}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
