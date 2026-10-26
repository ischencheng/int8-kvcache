#!/bin/bash
# Batch-1 ngram spec decode on 7B: bf16 FA2, INT8 with the current kernels, INT8 with the kernels before
# the split-KV merge rewrite (scripts/_prev_kernel/int8_kv.cu = git show 596c408:kvq/cuda/int8_kv.cu),
# interleaved for 3 rounds in one container (run-to-run drift is as large as the effect).
set -x
S=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
INT8='"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}'
OURS="--env KVQ_SCALES=$S --env KVQ_KERNEL=cuda --env KVQ_CFG=B_channel_smooth"
NG='"speculative_config":{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
for r in 1 2 3; do
  python3 scripts/bench_spec_latency.py --label ng7_bf16_r$r --llm "{$NG}"
  python3 scripts/bench_spec_latency.py --label ng7_int8_r$r $OURS --llm "{$INT8,$NG}"
  python3 scripts/bench_spec_latency.py --label ng7_int8prev_r$r $OURS --env KVQ_CUDA_SRC=/root/repo/scripts/_prev_kernel/int8_kv.cu --llm "{$INT8,$NG}"
done
