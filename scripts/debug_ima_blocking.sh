#!/bin/bash
# Find the faulting launch of the mixed-batch illegal memory access: synchronous launches, eager mode.
set -x
S=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
INT8='"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}'
python3 scripts/run_eval.py --label dbg_blocking --n 400 --env KVQ_SCALES=$S --env KVQ_KERNEL=cuda \
  --env KVQ_CFG=B_channel_smooth --env KVQ_DEBUG=1 --env CUDA_LAUNCH_BLOCKING=1 --env KVQ_DEBUG_DUMP=$OUT/fa_call.pt --llm "{$INT8,\"enforce_eager\":true}" \
  > $OUT/dbg_blocking.log 2>&1
grep -v triton_bundler $OUT/dbg_blocking.log | grep -E -B 2 -A 40 'Traceback|illegal' | head -n 300
