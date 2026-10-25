#!/bin/bash
# Advanced item: INT8 latent cache for MLA (DeepSeek-V2-Lite-Chat on A100, TRITON_MLA backend).
# Analysis + decode-op error on real dumps, then end-to-end accuracy with the cache fake-quantized at write
# time (= what an INT8 latent cache stores), plain vs smooth (asymmetric) scales, with / without INT8 k_pe.
set -x
M=deepseek-ai/DeepSeek-V2-Lite-Chat
D=/cache/mla/dsv2-lite
python3 scripts/mla_capture.py --model $M --n 256
python3 scripts/mla_analyze.py
LLM='{"trust_remote_code":true,"attention_config":{"backend":"TRITON_MLA"},"enforce_eager":true}'
python3 scripts/run_eval.py --model $M --label mla_bf16 --task gsm8k --max-tokens 1024 --llm "$LLM"
python3 scripts/run_eval.py --model $M --label mla_bf16 --task aime24 --max-tokens 2048 --llm "$LLM"
REFS="$OUT/eval_mla_bf16_gsm8k.jsonl $OUT/eval_mla_bf16_aime24.jsonl"
python3 scripts/eval_nll.py --model $M --refs $REFS --label mla_bf16 --llm "$LLM"
for v in "mla_latent_plain scales 0" "mla_latent_smooth scales_smooth 0" "mla_all_smooth scales_smooth 1"; do
  set -- $v
  E="--env KVQ_MLA=fakequant --env KVQ_MLA_SCALES=$D/$2.safetensors --env KVQ_MLA_ROPE=$3"
  python3 scripts/run_eval.py --model $M --label $1 --task gsm8k --max-tokens 1024 $E --llm "$LLM"
  python3 scripts/compare_runs.py $OUT/eval_mla_bf16_gsm8k.jsonl $OUT/eval_$1_gsm8k.jsonl > $OUT/compare_mla_bf16_$1.json
  python3 scripts/eval_nll.py --model $M --refs $REFS --label $1 $E --llm "$LLM"
done
