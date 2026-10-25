#!/bin/bash
set -x
python3 scripts/calibrate.py --calib math --n 64
S=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
EAGER='{"enforce_eager":true}'
INT8='{"enforce_eager":true,"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}}'
python3 scripts/run_eval.py --label int8_static_torch --n 50 --env KVQ_SCALES=$S --env KVQ_CFG=B_channel_smooth --llm "$INT8"
python3 scripts/run_eval.py --label bisect_static --n 50 --env KVQ_SCALES=$S --env KVQ_ATTN=static:B_channel_smooth --llm "$EAGER"
python3 scripts/compare_runs.py $OUT/eval_bisect_static_gsm8k.jsonl $OUT/eval_int8_static_torch_gsm8k.jsonl
