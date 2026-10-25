"""MLA: capture latent-cache statistics and decode dumps on DeepSeek-V2-Lite, write static scales.

Calibration prompts are GSM8K train (answered by the model); evaluation uses GSM8K test.
"""
import argparse
import json
import os
import random

os.environ["KVQ_MLA"] = "capture"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from datasets import load_dataset
from safetensors.torch import save_file
from vllm import LLM, SamplingParams

from kvq import data, mla, quant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = random.Random(1).sample(list(ds), args.n)
    llm = LLM(args.model, enforce_eager=True, max_model_len=4096, gpu_memory_utilization=0.85, seed=0,
              trust_remote_code=True, attention_config={"backend": "TRITON_MLA"})
    llm.chat([data.math_messages(r["question"]) for r in rows], SamplingParams(temperature=0, max_tokens=1024))
    cap = llm.collective_rpc(mla.collect)[0]
    root = "/cache/mla/dsv2-lite"
    os.makedirs(root, exist_ok=True)
    torch.save({"stats": cap["stats"], "samples": cap["samples"]}, f"{root}/capture.pt")
    flat = {f"{i}|{k}": v.contiguous() for i, d in cap["dumps"].items() for k, v in d.items()}
    if flat:
        save_file(flat, f"{root}/decode_dumps.safetensors")
    # Static per-channel scales: plain (mu = 0) and smooth (asymmetric around the channel mean).
    for smooth in (False, True):
        out = {}
        for i, st in cap["stats"].items():
            c = st["c"]
            mu = (c["sum"] / c["tokens"]).float() if smooth else torch.zeros_like(c["absmax"])
            amax = torch.maximum(c["max"] - mu, mu - c["min"]) if smooth else c["absmax"]
            out[f"{i}|c_scale"] = quant.scale_from_absmax(amax)
            out[f"{i}|c_mu"] = mu
            out[f"{i}|kpe_scale"] = quant.scale_from_absmax(st["kpe"]["absmax"])
        save_file({k: v.contiguous().float() for k, v in out.items()},
                  f"{root}/scales{'_smooth' if smooth else ''}.safetensors")
    summary = {"layers": len(cap["stats"]), "dumps": sorted(cap["dumps"]),
               "tokens": cap["stats"][0]["c"]["tokens"]}
    print(json.dumps(summary))
    with open(f"{os.environ['OUT']}/mla_capture.json", "w") as f:
        json.dump(summary, f)


if __name__ == "__main__":
    main()
