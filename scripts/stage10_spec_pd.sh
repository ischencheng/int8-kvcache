#!/bin/bash
# CUDA-only backend: spec-decode acceptance (verification batches on the INT8 kernel by default, *_sq1
# through dequantize + FlashAttention) and PD disaggregation (producer prefill is now FlashAttention).
set -x
T=Qwen/Qwen2.5-1.5B-Instruct
ST=/cache/scales/qwen2.5-1.5b/math_per_channel_max_smooth.safetensors
SD=/cache/scales/qwen2.5-0.5b/math_per_channel_max_smooth.safetensors
S7=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
DRAFT='"method":"draft_model","model":"Qwen/Qwen2.5-0.5B-Instruct","num_speculative_tokens":4'
NG='"speculative_config":{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
INT8='"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}'
E15="--env KVQ_SCALES=$ST --env KVQ_CFG=B_channel_smooth --env KVQ_KERNEL=cuda --env KVQ_DRAFT_SCALES=$SD"
SPEC_INT8="{$INT8,\"speculative_config\":{$DRAFT,\"attention_backend\":\"CUSTOM\"}}"
python3 scripts/run_eval.py --model $T --label sd_bf16_bf16 --task gsm8k --n 500 --llm "{\"speculative_config\":{$DRAFT}}"
python3 scripts/run_eval.py --model $T --label sd_int8_int8 --task gsm8k --n 500 $E15 --llm "$SPEC_INT8"
python3 scripts/run_eval.py --model $T --label sd_int8_int8_sq1 --task gsm8k --n 500 $E15 --env KVQ_SHORT_Q=1 --llm "$SPEC_INT8"
python3 scripts/run_eval.py --label ng_int8 --task gsm8k --n 500 \
  --env KVQ_SCALES=$S7 --env KVQ_CFG=B_channel_smooth --env KVQ_KERNEL=cuda --llm "{$INT8,$NG}"
bash scripts/ext_pd.sh
