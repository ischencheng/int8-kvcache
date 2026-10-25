"""Stage 6 / extension 3.5(a): collect KV statistics on a calibration set and write static scales.

    python3 scripts/calibrate.py --calib math --tp 1
writes /cache/scales/<tag>/<calib>[_tp<N>].stats.pt and one scales file per
(granularity, method, smooth) variant under /cache/scales/<tag>/ and $OUT/scales/.
Calibration never uses the evaluation splits (GSM8K test, AIME24).
"""
import argparse
import json
import os
import random

os.environ["KVQ_CAPTURE"] = "1"
os.environ.setdefault("KVQ_CAPTURE_MAX_TOKENS", "8192")

import torch
from datasets import load_dataset
from transformers import AutoConfig
from vllm import LLM, SamplingParams

from kvq import capture, data, scales

VARIANTS = [  # (granularity, method, smooth_k)
    ("per_channel", "max", False), ("per_channel", "max", True), ("per_channel", "q0.999", True),
    ("per_channel", "q0.9999", True), ("per_head", "max", False), ("per_head", "max", True),
]


def calib_prompts(kind, n):
    if kind == "math":   # GSM8K train questions, answered by the model (prefill + decode KV)
        ds = load_dataset("openai/gsm8k", "main", split="train")
        rows = random.Random(1).sample(list(ds), n)
        return [data.math_messages(r["question"]) for r in rows], 512
    if kind == "text":   # raw Wikipedia text, prefill only
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(t for t in ds["text"] if t.strip())
        chunks = [text[i:i + 6000] for i in range(0, len(text), 6000)]
        rows = random.Random(1).sample(chunks, n)
        return [[{"role": "user", "content": f"Summarize:\n{c}"}] for c in rows], 1
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tag", default="qwen2.5-7b")
    ap.add_argument("--calib", default="math", choices=["math", "text"])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args()

    msgs, max_tokens = calib_prompts(args.calib, args.n)
    if args.tp == 1:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # no IPC for the sampled rows
    llm = LLM(args.model, enforce_eager=True, max_model_len=4096, gpu_memory_utilization=0.85, seed=0,
              tensor_parallel_size=args.tp)
    num_kv_heads = AutoConfig.from_pretrained(args.model).num_key_value_heads
    llm.chat(msgs, SamplingParams(temperature=0, max_tokens=max_tokens))
    per_rank = llm.collective_rpc(capture.collect)
    stats, samples = scales.merge_tp(per_rank, num_kv_heads)

    suffix = f"{args.calib}" + (f"_tp{args.tp}" if args.tp > 1 else "")
    root = f"/cache/scales/{args.tag}"
    out_dir = f"{os.environ['OUT']}/scales"
    os.makedirs(root, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"stats": stats}, f"{root}/{suffix}.stats.pt")
    summary = {"tokens": next(iter(stats.values()))["tokens"], "variants": []}
    for gran, method, smooth in VARIANTS:
        layer_scales = {name: scales.from_stats(stats[name], samples[name], gran, method, smooth) for name in stats}
        fname = f"{suffix}_{gran}_{method}{'_smooth' if smooth else ''}.safetensors"
        scales.save(f"{root}/{fname}", layer_scales)
        scales.save(f"{out_dir}/{fname}", layer_scales)
        summary["variants"].append(fname)
    print(json.dumps(summary, indent=1))
    with open(f"{os.environ['OUT']}/calib_{suffix}.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
