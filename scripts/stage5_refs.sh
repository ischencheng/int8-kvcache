#!/bin/bash
# bf16 FA2 reference answers (accuracy baseline + fixed texts for the NLL metric), saved to the volume.
set -x
EAGER='{"enforce_eager":true}'
R=/cache/refs/qwen2.5-7b && mkdir -p $R
python3 scripts/run_eval.py --label fa2 --task gsm8k --max-tokens 1024 --llm "$EAGER"
python3 scripts/run_eval.py --label fa2 --task aime24 --max-tokens 2048 --llm "$EAGER"
cp $OUT/eval_fa2_gsm8k.jsonl $OUT/eval_fa2_aime24.jsonl $R/
python3 scripts/eval_nll.py --refs $R/eval_fa2_gsm8k.jsonl $R/eval_fa2_aime24.jsonl --label fa2 --llm "$EAGER"
