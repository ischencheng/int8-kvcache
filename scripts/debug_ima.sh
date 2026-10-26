#!/bin/bash
# Regression check for the illegal memory access of the first 1319-question run (mixed prefill + decode
# batches: the long-query plan gathered page ids before the step's block table was in place) and for
# head_dim 64 (Qwen2.5-0.5B as an INT8 draft model).
set -x
S=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
INT8='"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}'
OURS="--env KVQ_SCALES=$S --env KVQ_KERNEL=cuda --env KVQ_CFG=B_channel_smooth"
python3 -m pytest -q tests
python3 scripts/run_eval.py --label dbg_check --n 400 $OURS --env KVQ_DEBUG=1 --llm "{$INT8}"
python3 scripts/run_eval.py --label dbg_graph --n 400 $OURS --llm "{$INT8}"
T=Qwen/Qwen2.5-1.5B-Instruct
ST=/cache/scales/qwen2.5-1.5b/math_per_channel_max_smooth.safetensors
SD=/cache/scales/qwen2.5-0.5b/math_per_channel_max_smooth.safetensors
DRAFT='"method":"draft_model","model":"Qwen/Qwen2.5-0.5B-Instruct","num_speculative_tokens":4'
python3 scripts/run_eval.py --model $T --label dbg_sd_int8_int8 --n 50 \
  --env KVQ_SCALES=$ST --env KVQ_CFG=B_channel_smooth --env KVQ_KERNEL=cuda --env KVQ_DRAFT_SCALES=$SD \
  --llm "{$INT8,\"speculative_config\":{$DRAFT,\"attention_backend\":\"CUSTOM\"}}"
