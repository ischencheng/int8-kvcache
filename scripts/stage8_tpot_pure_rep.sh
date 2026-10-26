#!/bin/bash
# Stage 8 follow-up: repeat two pure-decode points with a per-point warm-up (checks run-to-run noise / JIT in T(L)).
set -x
S=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
PTS="1024:128;4096:128"
INT8='"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}'
python3 scripts/bench_e2e.py --pure --repeat 3 --label fa2_bf16 --points "$PTS"
python3 scripts/bench_e2e.py --pure --repeat 3 --label ours_int8_cuda --points "$PTS" --env KVQ_SCALES=$S --env KVQ_KERNEL=cuda --env KVQ_CFG=B_channel_smooth --llm "{$INT8}"
