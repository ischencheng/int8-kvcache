#!/bin/bash
# Stage 5 worker: dynamic INT8 quantization end to end for the given configs (PyTorch attention, eager).
#   bash scripts/stage5_dynamic.sh A_head A_channel ...
set -x
EAGER='{"enforce_eager":true}'
R=/cache/refs/qwen2.5-7b
for m in "$@"; do
  python3 scripts/run_eval.py --label $m --task gsm8k --max-tokens 1024 --env KVQ_ATTN=$m --llm "$EAGER"
  python3 scripts/compare_runs.py $R/eval_fa2_gsm8k.jsonl $OUT/eval_${m}_gsm8k.jsonl > $OUT/compare_fa2_${m}.json
  python3 scripts/eval_nll.py --refs $R/eval_fa2_gsm8k.jsonl $R/eval_fa2_aime24.jsonl --label $m --env KVQ_ATTN=$m --llm "$EAGER"
done
