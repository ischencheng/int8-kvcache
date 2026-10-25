#!/bin/bash
# Nsight Compute on the decode attention kernels (Modal: clocks cannot be locked -> --clock-control none).
set -x
NCU=$(ls /opt/nvidia/nsight-compute/*/ncu /usr/local/cuda/bin/ncu 2>/dev/null | head -1)
cd scripts
$NCU --clock-control none --profile-from-start off --target-processes all \
  --section SpeedOfLight --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
  --section Occupancy --section LaunchStats --section WarpStateStats \
  --metrics dram__bytes_read.sum,dram__bytes_read.sum.per_second,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_tensor_op_imma.sum,sm__inst_executed_pipe_tensor_op_hmma.sum \
  -o $OUT/decode -f python3 ncu_decode.py
$NCU --import $OUT/decode.ncu-rep --csv --page raw > $OUT/decode_raw.csv
$NCU --import $OUT/decode.ncu-rep --page details > $OUT/decode_details.txt
