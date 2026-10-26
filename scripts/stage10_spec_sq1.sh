#!/bin/bash
# Rerun of the INT8 draft-model runs after the fix (FlashAttention key counts from the exact GPU lengths;
# vLLM's host-side lengths are upper bounds under async spec decode, which had lowered the draft model's
# prompt attention and the acceptance rate): default routing, and *_sq1 with the verification batches
# through dequantize + FlashAttention.
set -x
T=Qwen/Qwen2.5-1.5B-Instruct
ST=/cache/scales/qwen2.5-1.5b/math_per_channel_max_smooth.safetensors
SD=/cache/scales/qwen2.5-0.5b/math_per_channel_max_smooth.safetensors
DRAFT='"method":"draft_model","model":"Qwen/Qwen2.5-0.5B-Instruct","num_speculative_tokens":4'
INT8='"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}'
python3 -m pytest -q tests
python3 scripts/run_eval.py --model $T --label sd_int8_int8 --task gsm8k --n 500 \
  --env KVQ_SCALES=$ST --env KVQ_CFG=B_channel_smooth --env KVQ_KERNEL=cuda --env KVQ_DRAFT_SCALES=$SD \
  --llm "{$INT8,\"speculative_config\":{$DRAFT,\"attention_backend\":\"CUSTOM\"}}"
python3 scripts/run_eval.py --model $T --label sd_int8_int8_sq1 --task gsm8k --n 500 \
  --env KVQ_SCALES=$ST --env KVQ_CFG=B_channel_smooth --env KVQ_KERNEL=cuda --env KVQ_DRAFT_SCALES=$SD --env KVQ_SHORT_Q=1 \
  --llm "{$INT8,\"speculative_config\":{$DRAFT,\"attention_backend\":\"CUSTOM\"}}"
