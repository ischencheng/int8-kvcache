#!/bin/bash
# Extension 3.5(d): PD disaggregation (NixlConnector, prefill on GPU0, decode on GPU1) vs colocated,
# bf16 KV vs static INT8 KV. Accuracy through the proxy, then TTFT/TPOT under load with vllm bench serve.
set -x
M=Qwen/Qwen2.5-7B-Instruct
S=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
export KVQ_CFG=B_channel_smooth KVQ_KERNEL=triton KVQ_SCALES=$S
KVT='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
INT8_ARGS='--kv-cache-dtype int8_static --attention-backend CUSTOM'
COMMON="--max-model-len 8192 --gpu-memory-utilization 0.85"

wait_up() { for i in $(seq 1 180); do curl -sf localhost:$1/v1/models >/dev/null && return 0; sleep 2; done; return 1; }
bench() {  # label url concurrency
  vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url $2 --model $M \
    --dataset-name random --random-input-len 2048 --random-output-len 256 --num-prompts $(( $3 * 4 )) \
    --max-concurrency $3 --ignore-eos --save-result --result-dir $OUT --result-filename bench_$1_c$3.json \
    --percentile-metrics ttft,tpot,itl,e2el 2>&1 | tail -30
}
run_pd() {  # label extra-args
  CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 vllm serve $M --port 8100 $COMMON $2 \
    --kv-transfer-config "$KVT" > $OUT/log_$1_prefill.txt 2>&1 &
  CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 vllm serve $M --port 8200 $COMMON $2 \
    --kv-transfer-config "$KVT" > $OUT/log_$1_decode.txt 2>&1 &
  wait_up 8100 && wait_up 8200 || { tail -50 $OUT/log_$1_*.txt; return 1; }
  python3 scripts/pd_proxy.py --port 8000 > $OUT/log_$1_proxy.txt 2>&1 &
  sleep 5
  python3 scripts/eval_api.py --url http://localhost:8000 --label $1 --n 500
  for c in 32 128; do bench $1 http://localhost:8000 $c; done
  grep -h "GPU KV cache size" $OUT/log_$1_*.txt
  pkill -f "vllm serve"; pkill -f pd_proxy.py; sleep 10
}
run_colo() {  # label extra-args
  CUDA_VISIBLE_DEVICES=0 vllm serve $M --port 8100 $COMMON $2 > $OUT/log_$1.txt 2>&1 &
  wait_up 8100 || { tail -50 $OUT/log_$1.txt; return 1; }
  python3 scripts/eval_api.py --url http://localhost:8100 --label $1 --n 500
  for c in 32 128; do bench $1 http://localhost:8100 $c; done
  pkill -f "vllm serve"; sleep 10
}
run_pd pd_bf16 ""
run_pd pd_int8 "$INT8_ARGS"
run_colo colo_bf16 ""
run_colo colo_int8 "$INT8_ARGS"
