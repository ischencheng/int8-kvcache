#!/bin/bash
# CUDA-only backend smoke test: unit tests, a short GSM8K run with CUDA graphs, ngram spec decode
# (uniform verification batches of 5 tokens in full graphs) and one long prefill (dequantize + FA2).
set -x
S=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
INT8='"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}'
OURS="--env KVQ_SCALES=$S --env KVQ_KERNEL=cuda --env KVQ_CFG=B_channel_smooth"
NG='"speculative_config":{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
python3 -m pytest -q tests
python3 scripts/run_eval.py --label smoke_int8 --n 50 $OURS --llm "{$INT8}"
python3 scripts/run_eval.py --label smoke_int8_ngram --n 20 $OURS --llm "{$INT8,$NG}"
python3 scripts/bench_e2e.py --label smoke_int8 --points "16384:2" --out-len 8 $OURS --llm "{$INT8}"
