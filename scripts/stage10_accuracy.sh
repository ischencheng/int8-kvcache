#!/bin/bash
# CUDA-only backend: GSM8K (1319, CUDA graphs) and teacher-forced NLL, with the default routing (long
# prompts dequantize + FlashAttention) and with every row on the INT8 kernel (decode numerics).
set -x
S=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
R=/cache/refs/qwen2.5-7b
INT8='"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}'
OURS="--env KVQ_SCALES=$S --env KVQ_KERNEL=cuda --env KVQ_CFG=B_channel_smooth"
REFS="$R/eval_fa2_gsm8k.jsonl $R/eval_fa2_aime24.jsonl"
python3 scripts/run_eval.py --label ours_int8 --task gsm8k --max-tokens 1024 $OURS --llm "{$INT8}"
python3 scripts/compare_runs.py $R/eval_fa2_gsm8k.jsonl $OUT/eval_ours_int8_gsm8k.jsonl > $OUT/compare_fa2_ours_int8.json
python3 scripts/eval_nll.py --refs $REFS --label fa2
python3 scripts/eval_nll.py --refs $REFS --label ours_int8 $OURS --llm "{$INT8}"
python3 scripts/eval_nll.py --refs $REFS --label ours_int8_allshort $OURS --env KVQ_SHORT_Q=1000000 \
  --llm "{$INT8,\"enforce_eager\":true}"
