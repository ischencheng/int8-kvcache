#!/bin/bash
# FA2 out-of-bounds softmax_lse write for rows with seqused_k = 0 (docs/LOG.md, Step 10): every variant plain,
# then compute-sanitizer memcheck with the caching allocator off so it sees each tensor's bounds.
set -x
nvidia-smi --query-gpu=name,driver_version --format=csv
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
if [ "$FA_IMPL" = upstream ]; then python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
else python3 -c "import vllm; print('vllm', vllm.__version__)"; fi
CS=$(command -v compute-sanitizer || find /usr/local -name compute-sanitizer -type f 2>/dev/null | head -1)
if [ -z "$CS" ]; then
  V=$(python3 -c "import torch; print(torch.version.cuda.replace('.', '-'))")
  apt-get update -qq && apt-get install -y -qq "cuda-sanitizer-$V" > /dev/null
  CS=$(find /usr/local -name compute-sanitizer -type f | head -1)
fi
$CS --version
P=scripts/repro_fa2_empty_rows.py
python3 $P
MC="$CS --tool memcheck --print-limit 100000 --show-backtrace device"
for v in serving serving_control serving_low serving_high small small_control big big_control; do
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 $MC python3 $P $v > $OUT/memcheck_$v.log 2>&1
  python3 $P --check $v $OUT/memcheck_$v.log
done
# caching allocator on: the small batch's overflow stays inside PyTorch's pool, memcheck sees nothing
$MC python3 $P small > $OUT/memcheck_small_caching.log 2>&1
python3 $P --check small $OUT/memcheck_small_caching.log
