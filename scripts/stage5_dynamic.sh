#!/bin/bash
# Stage 5: dynamic INT8 quantization end to end (PyTorch attention inside vLLM, eager).
set -x
EAGER='{"enforce_eager":true}'
python3 scripts/run_eval.py --label fa2 --task gsm8k --max-tokens 1024 --llm "$EAGER"
python3 scripts/run_eval.py --label fa2 --task aime24 --max-tokens 2048 --llm "$EAGER"
REFS="$OUT/eval_fa2_gsm8k.jsonl $OUT/eval_fa2_aime24.jsonl"
python3 scripts/eval_nll.py --refs $REFS --label fa2 --llm "$EAGER"
for m in A_head A_token_head A_channel A_channel_smooth B_head B_channel B_channel_smooth B_channel_smooth_qgroup32; do
  python3 scripts/run_eval.py --label $m --task gsm8k --max-tokens 1024 --env KVQ_ATTN=$m --llm "$EAGER"
  python3 scripts/compare_runs.py $OUT/eval_fa2_gsm8k.jsonl $OUT/eval_${m}_gsm8k.jsonl > $OUT/compare_fa2_${m}.json
  python3 scripts/eval_nll.py --refs $REFS --label $m --env KVQ_ATTN=$m --llm "$EAGER"
done
