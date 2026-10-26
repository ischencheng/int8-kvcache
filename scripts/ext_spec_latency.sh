#!/bin/bash
# Extension 3.5(c), speed part: batch-1 TPOT with real prompts, with / without spec decode, bf16 / INT8 KV.
set -x
T=Qwen/Qwen2.5-1.5B-Instruct
ST=/cache/scales/qwen2.5-1.5b/math_per_channel_max_smooth.safetensors
SD=/cache/scales/qwen2.5-0.5b/math_per_channel_max_smooth.safetensors
S7=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
DRAFT='"method":"draft_model","model":"Qwen/Qwen2.5-0.5B-Instruct","num_speculative_tokens":4'
NG='"speculative_config":{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
INT8='"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}'
E15="--env KVQ_SCALES=$ST --env KVQ_CFG=B_channel_smooth --env KVQ_KERNEL=cuda --env KVQ_DRAFT_SCALES=$SD"
E7="--env KVQ_SCALES=$S7 --env KVQ_CFG=B_channel_smooth --env KVQ_KERNEL=cuda"
python3 scripts/bench_spec_latency.py --model $T --label sd_none_bf16
python3 scripts/bench_spec_latency.py --model $T --label sd_none_int8 $E15 --llm "{$INT8}"
python3 scripts/bench_spec_latency.py --model $T --label sd_bf16_bf16 --llm "{\"speculative_config\":{$DRAFT}}"
python3 scripts/bench_spec_latency.py --model $T --label sd_int8_bf16 $E15 --llm "{$INT8,\"speculative_config\":{$DRAFT,\"kv_cache_dtype\":\"auto\"}}"
python3 scripts/bench_spec_latency.py --model $T --label sd_int8_int8 $E15 --llm "{$INT8,\"speculative_config\":{$DRAFT,\"attention_backend\":\"CUSTOM\"}}"
python3 scripts/bench_spec_latency.py --label ng7_none_bf16
python3 scripts/bench_spec_latency.py --label ng7_none_int8 $E7 --llm "{$INT8}"
python3 scripts/bench_spec_latency.py --label ng7_bf16 --llm "{$NG}"
python3 scripts/bench_spec_latency.py --label ng7_int8 $E7 --llm "{$INT8,$NG}"
