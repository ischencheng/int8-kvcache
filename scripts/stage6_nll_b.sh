#!/bin/bash
# Stage 6 follow-up: teacher-forced NLL with every row on the INT8-arithmetic kernel, so the metric
# covers the decode numerics too (normally long prefill rows dequantize). The group32 q-scale row ran
# with the Triton kernel (tag triton-backend); the CUDA kernel has one q scale per (token, head).
set -x
R=/cache/refs/qwen2.5-7b
S=/cache/scales/qwen2.5-7b
INT8='{"enforce_eager":true,"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}}'
for v in "st_ch_max math_per_channel_max B_channel" "st_ch_max_smooth math_per_channel_max_smooth B_channel_smooth" \
         "st_head_max_smooth math_per_head_max_smooth B_head_smooth" \
         "st_text_ch_smooth text_per_channel_max_smooth B_channel_smooth"; do
  set -- $v
  python3 scripts/eval_nll.py --refs $R/eval_fa2_gsm8k.jsonl $R/eval_fa2_aime24.jsonl --label nllB_$1 \
    --env KVQ_SCALES=$S/$2.safetensors --env KVQ_CFG=$3 --env KVQ_KERNEL=cuda --env KVQ_SHORT_Q=1000000 --llm "$INT8"
done
