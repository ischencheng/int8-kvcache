"""Stage 2: run real prompts and capture what goes into the KV cache.

Writes the full capture (stats, sampled K/V rows, decode dumps) to the volume
under /cache/kv_capture/<tag>/ and small summaries to $OUT.
"""
import argparse
import json
import os

os.environ["KVQ_CAPTURE"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # keep the worker in-process

import torch
from safetensors.torch import save_file
from vllm import LLM, SamplingParams

from kvq import capture, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tag", default="qwen2.5-7b")
    ap.add_argument("--gsm8k", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    rows = data.aime24() + data.gsm8k(args.gsm8k)
    # Python hooks inside attention only run in eager mode: a full CUDA graph
    # replays the kernels without calling back into Python.
    llm = LLM(args.model, enforce_eager=True, max_model_len=4096, gpu_memory_utilization=0.85, seed=0)
    outs = llm.chat([data.math_messages(r["question"]) for r in rows],
                    SamplingParams(temperature=0, max_tokens=args.max_tokens))
    cap = llm.collective_rpc(capture.collect)[0]

    root = f"/cache/kv_capture/{args.tag}"
    os.makedirs(root, exist_ok=True)
    torch.save({"stats": cap["stats"], "samples": cap["samples"]}, f"{root}/capture.pt")
    flat = {f"{name}|{k}": v.contiguous() for name, d in cap["dumps"].items() for k, v in d.items()}
    if flat:
        save_file(flat, f"{root}/decode_dumps.safetensors")

    out = os.environ["OUT"]
    torch.save(cap["stats"], f"{out}/stats_{args.tag}.pt")
    lens = [len(o.prompt_token_ids) + len(o.outputs[0].token_ids) for o in outs]
    summary = {"model": args.model, "num_prompts": len(rows), "total_tokens": sum(lens),
               "max_seq_len": max(lens), "dumped_layers": sorted(cap["dumps"]),
               "sampled_tokens_per_layer": {k: v["k"].shape[0] for k, v in list(cap["samples"].items())[:1]}}
    print(json.dumps(summary, indent=2))
    with open(f"{out}/capture_{args.tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(f"{out}/generations_{args.tag}.jsonl", "w") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"id": r["id"], "answer": r["answer"], "text": o.outputs[0].text}) + "\n")


if __name__ == "__main__":
    main()
