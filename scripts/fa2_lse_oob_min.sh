#!/bin/bash
# The issue's minimal repro on upstream flash-attn: plain, then memcheck with the caching allocator off.
set -x
python3 -c "import torch, flash_attn; print(torch.__version__, flash_attn.__version__)"
python3 scripts/fa2_lse_oob_min.py
CS=$(command -v compute-sanitizer || find /usr/local -name compute-sanitizer -type f | head -1)
PYTORCH_NO_CUDA_MEMORY_CACHING=1 $CS --tool memcheck python3 scripts/fa2_lse_oob_min.py > $OUT/memcheck_min.log 2>&1
head -n 40 $OUT/memcheck_min.log
tail -n 3 $OUT/memcheck_min.log
