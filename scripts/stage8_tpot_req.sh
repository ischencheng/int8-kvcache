#!/bin/bash
# Stage 8 follow-up: per-request TPOT (engine first/last token timestamps) on the key points.
set -x
S=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
PTS="1024:1,64;4096:16,64,128;16384:16"
INT8='"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}'
python3 scripts/bench_e2e.py --label fa2_bf16 --points "$PTS"
python3 scripts/bench_e2e.py --label ours_int8_cuda --points "$PTS" --env KVQ_SCALES=$S --env KVQ_KERNEL=cuda --env KVQ_CFG=B_channel_smooth --llm "{$INT8}"
python3 scripts/bench_e2e.py --label vllm_int8_pth --points "$PTS" --llm '{"kv_cache_dtype":"int8_per_token_head","attention_config":{"backend":"TRITON_ATTN"}}'
