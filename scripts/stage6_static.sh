#!/bin/bash
# Stage 6 + extension 3.5(a): calibrate, then static INT8 KV end to end (int8_static + CUSTOM backend).
set -x
R=/cache/refs/qwen2.5-7b
S=/cache/scales/qwen2.5-7b
python3 scripts/calibrate.py --calib math --n 512
python3 scripts/calibrate.py --calib text --n 256
INT8='{"enforce_eager":true,"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}}'
run() {  # label scales cfg kernel
  local label=$1 scales=$2 cfg=$3 kernel=$4
  local env="--env KVQ_SCALES=$S/$scales.safetensors --env KVQ_CFG=$cfg --env KVQ_KERNEL=$kernel"
  python3 scripts/run_eval.py --label $label --task gsm8k --max-tokens 1024 $env --llm "$INT8"
  python3 scripts/compare_runs.py $R/eval_fa2_gsm8k.jsonl $OUT/eval_${label}_gsm8k.jsonl > $OUT/compare_fa2_${label}.json
  python3 scripts/eval_nll.py --refs $R/eval_fa2_gsm8k.jsonl $R/eval_fa2_aime24.jsonl --label $label $env --llm "$INT8"
}
run st_ch_max            math_per_channel_max              B_channel                 triton
run st_ch_max_smooth     math_per_channel_max_smooth       B_channel_smooth          triton
run st_ch_q999_smooth    math_per_channel_q0.999_smooth    B_channel_smooth          triton
run st_ch_q9999_smooth   math_per_channel_q0.9999_smooth   B_channel_smooth          triton
run st_head_max_smooth   math_per_head_max_smooth          B_head_smooth             triton
run st_text_ch_smooth    text_per_channel_max_smooth       B_channel_smooth          triton
run st_ch_smooth_g32     math_per_channel_max_smooth       B_channel_smooth_qgroup32 triton
run st_ch_smooth_A       math_per_channel_max_smooth       A_channel_smooth          triton
# torch path (stage-6 PyTorch implementation) on the main config, to tie the kernels to it end to end
python3 scripts/run_eval.py --label st_ch_max_smooth_torch --task gsm8k --n 250 --max-tokens 1024 \
  --env KVQ_SCALES=$S/math_per_channel_max_smooth.safetensors --env KVQ_CFG=B_channel_smooth --env KVQ_KERNEL=torch --llm "$INT8"
python3 scripts/run_eval.py --label st_ch_max_smooth_triton250 --task gsm8k --n 250 --max-tokens 1024 \
  --env KVQ_SCALES=$S/math_per_channel_max_smooth.safetensors --env KVQ_CFG=B_channel_smooth --env KVQ_KERNEL=triton --llm "$INT8"
python3 scripts/compare_runs.py $OUT/eval_st_ch_max_smooth_torch_gsm8k.jsonl $OUT/eval_st_ch_max_smooth_triton250_gsm8k.jsonl > $OUT/compare_torch_triton.json
