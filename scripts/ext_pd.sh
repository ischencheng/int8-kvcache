#!/bin/bash
# Extension 3.5(d): PD disaggregation vs colocated, bf16 vs static INT8 KV (Qwen2.5-7B, GSM8K 500 at once).
# Two separate engines, one after the other on the same GPU: a prefill-only producer writes the KV pages through vLLM's ExampleConnector, a decode
# consumer loads them instead of recomputing the prompt. (Modal's gVisor sandbox has no /sys/class/net, so
# UCX/NixlConnector cannot initialize; an online proxy setup also reset connections under load.)
set -x
S=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
INT8="--env KVQ_SCALES=$S --env KVQ_CFG=B_channel_smooth --env KVQ_KERNEL=cuda"
INT8_LLM='{"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}}'
for kv in bf16 int8; do
  rm -rf /tmp/kvstore && mkdir -p /tmp/kvstore
  if [ $kv = int8 ]; then E="$INT8"; L="$INT8_LLM"; else E=""; L='{}'; fi
  CUDA_VISIBLE_DEVICES=0 python3 scripts/pd_offline.py --role producer --label $kv $E --llm "$L"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/pd_offline.py --role consumer --label $kv $E --llm "$L"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/pd_offline.py --role colocated --label $kv $E --llm "$L"
done
