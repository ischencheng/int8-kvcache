"""Stage 1: environment + baseline + find the decode PagedAttention call site.

Runs Qwen2.5-1.5B-Instruct once with the interfering features off
(eager, no chunked prefill, no prefix caching, max_model_len 256) and reports
which attention backend/impl each layer uses and how the per-layer KV cache
tensor is laid out.
"""
import json
import os
import subprocess

import torch
import triton
import vllm
from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def describe_worker(worker):
    runner = worker.model_runner
    ctx = runner.compilation_config.static_forward_context
    name, layer = next(iter(ctx.items()))
    kv = layer.kv_cache
    impl = layer.impl
    return {
        "layer": name,
        "backend": layer.attn_backend.get_name(),
        "impl": f"{type(impl).__module__}.{type(impl).__qualname__}",
        "num_heads": impl.num_heads, "num_kv_heads": impl.num_kv_heads, "head_size": impl.head_size,
        "kv_cache_dtype": impl.kv_cache_dtype,
        "kv_cache_shape": list(kv.shape), "kv_cache_stride": list(kv.stride()), "kv_cache_torch_dtype": str(kv.dtype),
        "num_layers": len(ctx),
    }


def main():
    info = {
        "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__, "triton": triton.__version__, "vllm": vllm.__version__,
        "cuda": torch.version.cuda,
    }
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    llm = LLM(MODEL, enforce_eager=True, enable_chunked_prefill=False, enable_prefix_caching=False,
              max_model_len=256, gpu_memory_utilization=0.5, seed=0)
    out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=16, temperature=0))
    info["sample"] = out[0].outputs[0].text
    info["worker"] = llm.collective_rpc(describe_worker)[0]
    print(json.dumps(info, indent=2))
    os.makedirs(os.environ["OUT"], exist_ok=True)
    with open(os.path.join(os.environ["OUT"], "env.json"), "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
