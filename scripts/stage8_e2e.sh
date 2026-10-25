#!/bin/bash
# Stage 8 + extension 3.5(b): end-to-end TPOT / throughput with CUDA graphs on (vLLM default),
# and the INT8 backend's accuracy with graphs vs eager.
set -x
S=/cache/scales/qwen2.5-7b/math_per_channel_max_smooth.safetensors
R=/cache/refs/qwen2.5-7b
PTS="1024:1,16,64,128;4096:1,16,64,128;16384:1,16,32"
OURS="--env KVQ_SCALES=$S --env KVQ_KERNEL=triton"
INT8='"kv_cache_dtype":"int8_static","attention_config":{"backend":"CUSTOM"}'
declare -A LLM=(
  [fa2_bf16]='{}'
  [vllm_triton_bf16]='{"attention_config":{"backend":"TRITON_ATTN"}}'
  [vllm_int8_pth]='{"kv_cache_dtype":"int8_per_token_head","attention_config":{"backend":"TRITON_ATTN"}}'
  [ours_bf16]='{"attention_config":{"backend":"CUSTOM"}}'
  [ours_int8]="{$INT8}"
  [ours_int8_deq]="{$INT8}"
  [ours_int8_eager]="{$INT8,\"enforce_eager\":true}"
  [ours_int8_cuda]="{$INT8}"
)
declare -A ENV=(
  [ours_bf16]="--env KVQ_KERNEL=triton"
  [ours_int8]="$OURS --env KVQ_CFG=B_channel_smooth"
  [ours_int8_deq]="$OURS --env KVQ_CFG=A_channel_smooth"
  [ours_int8_eager]="$OURS --env KVQ_CFG=B_channel_smooth"
  [ours_int8_cuda]="--env KVQ_SCALES=$S --env KVQ_KERNEL=cuda --env KVQ_CFG=B_channel_smooth"
)
# accuracy with CUDA graphs (the stage-6 runs were eager)
python3 scripts/run_eval.py --label ours_int8_graph --task gsm8k --max-tokens 1024 ${ENV[ours_int8]} --llm "${LLM[ours_int8]}"
python3 scripts/compare_runs.py $R/eval_fa2_gsm8k.jsonl $OUT/eval_ours_int8_graph_gsm8k.jsonl > $OUT/compare_fa2_ours_int8_graph.json
python3 scripts/run_eval.py --label ours_int8_cuda --task gsm8k --max-tokens 1024 ${ENV[ours_int8_cuda]} --llm "${LLM[ours_int8_cuda]}"
python3 scripts/compare_runs.py $R/eval_fa2_gsm8k.jsonl $OUT/eval_ours_int8_cuda_gsm8k.jsonl > $OUT/compare_fa2_ours_int8_cuda.json
python3 scripts/run_eval.py --label vllm_int8_pth --task gsm8k --max-tokens 1024 --llm "${LLM[vllm_int8_pth]}"
python3 scripts/compare_runs.py $R/eval_fa2_gsm8k.jsonl $OUT/eval_vllm_int8_pth_gsm8k.jsonl > $OUT/compare_fa2_vllm_int8_pth.json
python3 scripts/eval_nll.py --refs $R/eval_fa2_gsm8k.jsonl $R/eval_fa2_aime24.jsonl --label vllm_int8_pth --llm "${LLM[vllm_int8_pth]}"
for label in fa2_bf16 vllm_triton_bf16 vllm_int8_pth ours_bf16 ours_int8 ours_int8_cuda ours_int8_deq ours_int8_eager; do
  python3 scripts/bench_e2e.py --label $label --points "$PTS" ${ENV[$label]} --llm "${LLM[$label]}"
done
for label in fa2_bf16 vllm_int8_pth ours_int8 ours_int8_cuda; do
  L=$(echo "${LLM[$label]}" | sed 's/}$//'); [ "$L" = "{" ] && SEP="" || SEP=","
  python3 scripts/bench_e2e.py --label $label --throughput 512 ${ENV[$label]} --llm "$L$SEP\"max_num_seqs\":512}"
done
