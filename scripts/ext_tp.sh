#!/bin/bash
# Extension 3.5(a), distributed: calibration and static scales under tensor parallelism (2 GPUs).
set -x
S=/cache/scales/qwen2.5-7b
python3 scripts/calibrate.py --calib math --n 512 --tp 2
python3 scripts/compare_scales.py $S/math_per_channel_max_smooth.safetensors $S/math_tp2_per_channel_max_smooth.safetensors
INT8='"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}'
ENV="--env KVQ_CFG=B_channel_smooth --env KVQ_KERNEL=cuda"
python3 scripts/run_eval.py --label tp2_bf16 --task gsm8k --max-tokens 1024 --llm '{"tensor_parallel_size":2}'
python3 scripts/run_eval.py --label tp2_int8_tp1scales --task gsm8k --max-tokens 1024 \
  $ENV --env KVQ_SCALES=$S/math_per_channel_max_smooth.safetensors --llm "{\"tensor_parallel_size\":2,$INT8}"
python3 scripts/run_eval.py --label tp2_int8_tp2scales --task gsm8k --max-tokens 1024 \
  $ENV --env KVQ_SCALES=$S/math_tp2_per_channel_max_smooth.safetensors --llm "{\"tensor_parallel_size\":2,$INT8}"
python3 scripts/compare_runs.py $OUT/eval_tp2_int8_tp1scales_gsm8k.jsonl $OUT/eval_tp2_int8_tp2scales_gsm8k.jsonl > $OUT/compare_tp_scales.json
