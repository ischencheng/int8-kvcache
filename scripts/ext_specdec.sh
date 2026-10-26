#!/bin/bash
# Extension 3.5(c): speculative decoding with an INT8 KV cache.
#  (1) draft_model: Qwen2.5-0.5B drafts for Qwen2.5-1.5B (k = 4; vLLM requires equal vocab sizes, and
#      Qwen2.5-7B's 152064 differs from the small models' 151936):
#      bf16 target + bf16 draft | int8 target + bf16 draft | int8 target + int8 draft
#      (verification batches, k + 1 query tokens, run the INT8 kernel; *_sq1 sends them through the
#      dequantize + FlashAttention path instead)
#  (2) ngram on Qwen2.5-7B (no draft model; only the target's verify batches see the INT8 cache)
set -x
python3 scripts/calibrate.py --model Qwen/Qwen2.5-1.5B-Instruct --tag qwen2.5-1.5b --calib math --n 512
python3 scripts/calibrate.py --model Qwen/Qwen2.5-0.5B-Instruct --tag qwen2.5-0.5b --calib math --n 512
T=Qwen/Qwen2.5-1.5B-Instruct
ST=/cache/scales/qwen2.5-1.5b/math_per_channel_max_smooth.safetensors
SD=/cache/scales/qwen2.5-0.5b/math_per_channel_max_smooth.safetensors
DRAFT='"method":"draft_model","model":"Qwen/Qwen2.5-0.5B-Instruct","num_speculative_tokens":4'
INT8='"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}'
ENV="--env KVQ_SCALES=$ST --env KVQ_CFG=B_channel_smooth --env KVQ_KERNEL=cuda --env KVQ_DRAFT_SCALES=$SD"
declare -A LLM=(
  [sd_none_bf16]="{}"
  [sd_none_int8]="{$INT8}"
  [sd_bf16_bf16]="{\"speculative_config\":{$DRAFT}}"
  [sd_int8_bf16]="{$INT8,\"speculative_config\":{$DRAFT,\"kv_cache_dtype\":\"auto\"}}"
  [sd_int8_int8]="{$INT8,\"speculative_config\":{$DRAFT,\"attention_backend\":\"CUSTOM\"}}"
  [sd_int8_int8_sq1]="{$INT8,\"speculative_config\":{$DRAFT,\"attention_backend\":\"CUSTOM\"}}"
)
for label in sd_none_bf16 sd_none_int8 sd_bf16_bf16 sd_int8_bf16 sd_int8_int8 sd_int8_int8_sq1; do
  E=""
  [[ $label == sd_*_int8* || $label == sd_int8* ]] && E="$ENV"
  [[ $label == sd_none_bf16 || $label == sd_bf16_bf16 ]] && E=""
  [[ $label == *sq1 ]] && E="$E --env KVQ_SHORT_Q=1"
  python3 scripts/run_eval.py --model $T --label $label --task gsm8k --n 500 --max-tokens 1024 $E --llm "${LLM[$label]}"
  python3 scripts/bench_e2e.py --model $T --label $label --ctx 1024,4096 --batch 1,16 $E --llm "${LLM[$label]}"
done
# ngram on the 7B target
S7=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
NG='"speculative_config":{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
python3 scripts/run_eval.py --label ng_bf16 --task gsm8k --n 500 --max-tokens 1024 --llm "{$NG}"
python3 scripts/run_eval.py --label ng_int8 --task gsm8k --n 500 --max-tokens 1024 \
  --env KVQ_SCALES=$S7 --env KVQ_CFG=B_channel_smooth --env KVQ_KERNEL=cuda --llm "{$INT8,$NG}"
python3 scripts/run_eval.py --label ng_int8_sq1 --task gsm8k --n 500 --max-tokens 1024 \
  --env KVQ_SCALES=$S7 --env KVQ_CFG=B_channel_smooth --env KVQ_KERNEL=cuda --env KVQ_SHORT_Q=1 --llm "{$INT8,$NG}"
