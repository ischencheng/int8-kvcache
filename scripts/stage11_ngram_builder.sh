#!/bin/bash
# Where the remaining ngram step-time gap comes from: vLLM's Triton backend in bf16 shares our metadata
# builder (ngram turns async scheduling off, so per-step CPU work is on the critical path); bf16 FA2 and
# our INT8 again for a second sample.
set -x
S=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
INT8='"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}'
OURS="--env KVQ_SCALES=$S --env KVQ_KERNEL=cuda --env KVQ_CFG=B_channel_smooth"
NG='"speculative_config":{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
python3 scripts/bench_spec_latency.py --label ng7_bf16_tritonattn --llm "{\"attention_config\":{\"backend\":\"TRITON_ATTN\"},$NG}"
python3 scripts/bench_spec_latency.py --label ng7_bf16 --llm "{$NG}"
python3 scripts/bench_spec_latency.py --label ng7_int8 $OURS --llm "{$INT8,$NG}"
python3 scripts/bench_spec_latency.py --label ng7_none_bf16_tritonattn --llm '{"attention_config":{"backend":"TRITON_ATTN"}}'
