#!/bin/bash
# Source-level stall sampling of the CUDA decode kernel (batch 1 split / batch 64 no split).
set -x
NCU=$(ls /opt/nvidia/nsight-compute/*/ncu | head -1)
cd scripts
cat > /tmp/ncu_case.py <<'PY'
import sys, torch
sys.path.insert(0, ".")
from bench_kernels import SCALE, decode_case, segm, static_for
from kvq import cuda_kernels as ck, quant
for batch, ctx in [(32, 4096), (64, 4096)]:
    q, k, v, cu, sl, bt = decode_case(batch, ctx)
    st = static_for(k, v)
    kq = quant.quantize(k.float() - st["k_mu"], st["k_scale"]); vq = quant.quantize(v, st["v_scale"])
    out = torch.empty_like(q)
    ck.decode_attention(q, kq, vq, out, cu, sl, bt, SCALE, st, segm=segm(batch)); torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    ck.decode_attention(q, kq, vq, out, cu, sl, bt, SCALE, st, segm=segm(batch)); torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
PY
$NCU --clock-control none --profile-from-start off --set full --import-source yes -o $OUT/cuda_hot -f python3 /tmp/ncu_case.py
$NCU --import $OUT/cuda_hot.ncu-rep --page details > $OUT/cuda_hot_details.txt
$NCU --import $OUT/cuda_hot.ncu-rep --page source --csv --print-source sass > $OUT/cuda_hot_sass.csv 2>&1 || true
