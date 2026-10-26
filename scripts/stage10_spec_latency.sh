#!/bin/bash
# Batch-1 spec-decode latency on one machine, after the exact-length fix (the draft model's prompt
# attention had used vLLM's optimistic host lengths): all eight runs again, which also gives the
# ngram pair a second sample.
set -x
S=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
ST=/cache/scales/qwen2.5-1.5b/math_per_channel_max_smooth.safetensors
SD=/cache/scales/qwen2.5-0.5b/math_per_channel_max_smooth.safetensors
INT8='"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}'
OURS="--env KVQ_SCALES=$S --env KVQ_KERNEL=cuda --env KVQ_CFG=B_channel_smooth"
E15="--env KVQ_SCALES=$ST --env KVQ_CFG=B_channel_smooth --env KVQ_KERNEL=cuda --env KVQ_DRAFT_SCALES=$SD"
T=Qwen/Qwen2.5-1.5B-Instruct
DRAFT='"method":"draft_model","model":"Qwen/Qwen2.5-0.5B-Instruct","num_speculative_tokens":4'
NG='"speculative_config":{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
python3 scripts/bench_spec_latency.py --label ng7_none_bf16
python3 scripts/bench_spec_latency.py --label ng7_none_int8 $OURS --llm "{$INT8}"
python3 scripts/bench_spec_latency.py --label ng7_bf16 --llm "{$NG}"
python3 scripts/bench_spec_latency.py --label ng7_int8 $OURS --llm "{$INT8,$NG}"
python3 scripts/bench_spec_latency.py --model $T --label sd_none_bf16
python3 scripts/bench_spec_latency.py --model $T --label sd_none_int8 $E15 --llm "{$INT8}"
python3 scripts/bench_spec_latency.py --model $T --label sd_bf16_bf16 --llm "{\"speculative_config\":{$DRAFT}}"
python3 scripts/bench_spec_latency.py --model $T --label sd_int8_int8 $E15 --llm "{$INT8,\"speculative_config\":{$DRAFT,\"attention_backend\":\"CUSTOM\"}}"
