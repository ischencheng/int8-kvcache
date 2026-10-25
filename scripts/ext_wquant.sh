#!/bin/bash
# Extension 3.5(e): INT8 KV on top of weight/activation quantized models (W8A8 INT8, W4A16).
set -x
INT8='"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}'
ENV="--env KVQ_CFG=B_channel_smooth --env KVQ_KERNEL=triton"
BF16_SCALES=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
for pair in "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8 w8a8" "RedHatAI/Qwen2.5-7B-Instruct-quantized.w4a16 w4a16"; do
  set -- $pair; M=$1; T=$2
  python3 scripts/calibrate.py --model $M --tag $T --calib math --n 512
  python3 scripts/run_eval.py --model $M --label ${T}_bf16kv --task gsm8k --max-tokens 1024
  python3 scripts/run_eval.py --model $M --label ${T}_int8kv --task gsm8k --max-tokens 1024 \
    $ENV --env KVQ_SCALES=/cache/scales/$T/math_per_channel_max_smooth.safetensors --llm "{$INT8}"
  python3 scripts/run_eval.py --model $M --label ${T}_int8kv_bf16scales --task gsm8k --max-tokens 1024 \
    $ENV --env KVQ_SCALES=$BF16_SCALES --llm "{$INT8}"
  python3 scripts/compare_scales.py $BF16_SCALES /cache/scales/$T/math_per_channel_max_smooth.safetensors
  mv $OUT/compare_scales.json $OUT/compare_scales_$T.json
done
