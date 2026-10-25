#!/bin/bash
# Extension 3.5(c): speculative decoding (draft_model: Qwen2.5-0.5B drafts for Qwen2.5-7B, k = 4).
#   bf16 target + bf16 draft | int8 target + bf16 draft | int8 target + int8 draft
set -x
S=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
python3 scripts/calibrate.py --model Qwen/Qwen2.5-0.5B-Instruct --tag qwen2.5-0.5b --calib math --n 512
SD=/cache/scales/qwen2.5-0.5b/math_per_channel_max_smooth.safetensors
DRAFT='"method":"draft_model","model":"Qwen/Qwen2.5-0.5B-Instruct","num_speculative_tokens":4'
INT8='"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}'
ENV="--env KVQ_SCALES=$S --env KVQ_CFG=B_channel_smooth --env KVQ_KERNEL=triton"
declare -A LLM=(
  [sd_bf16_bf16]="{\"speculative_config\":{$DRAFT}}"
  [sd_int8_bf16]="{$INT8,\"speculative_config\":{$DRAFT,\"kv_cache_dtype\":\"auto\"}}"
  [sd_int8_int8]="{$INT8,\"speculative_config\":{$DRAFT,\"attention_backend\":\"CUSTOM\"}}"
  [sd_int8_int8_mq8]="{$INT8,\"speculative_config\":{$DRAFT,\"attention_backend\":\"CUSTOM\"}}"
)
for label in sd_bf16_bf16 sd_int8_bf16 sd_int8_int8 sd_int8_int8_mq8; do
  E=""
  [[ $label == sd_int8* ]] && E="$ENV --env KVQ_DRAFT_SCALES=$SD"
  [[ $label == *mq8 ]] && E="$E --env KVQ_INT8_MAX_Q=8"
  python3 scripts/run_eval.py --label $label --task gsm8k --n 500 --max-tokens 1024 $E --llm "${LLM[$label]}"
  python3 scripts/bench_e2e.py --label $label --ctx 1024,4096 --batch 1,16 $E --llm "${LLM[$label]}"
done
