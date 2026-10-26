#!/bin/bash
# Batch-1 ngram spec-decode latency on 7B, bf16 vs INT8, with and without spec decode: TPOT, acceptance
# and time per engine step.
set -x
S=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
INT8='"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}'
OURS="--env KVQ_SCALES=$S --env KVQ_KERNEL=cuda --env KVQ_CFG=B_channel_smooth"
NG='"speculative_config":{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
python3 scripts/bench_spec_latency.py --label ng7_none_bf16
python3 scripts/bench_spec_latency.py --label ng7_none_int8 $OURS --llm "{$INT8}"
python3 scripts/bench_spec_latency.py --label ng7_bf16 --llm "{$NG}"
python3 scripts/bench_spec_latency.py --label ng7_int8 $OURS --llm "{$INT8,$NG}"
