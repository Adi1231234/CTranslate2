#!/bin/bash
# Nsight Compute on the production runner (an entry.sh "script" line, privileged job): for each heavy kernel, one
# runner process on the first units of a list with that kernel's launches past about <skip> decoding steps
# measured (time, DRAM, L2 and SM throughput, occupancy, where the warps stall), the process ended as soon as they
# are (--kill). One process per kernel, so each kernel gets its own sample of the steady state (one filter for all
# would sample mostly the most frequent one). Each report and its CSV go to the job's results as soon as they exist
# (a job past its time limit uploads nothing at the end). Nsight Compute is installed here from NVIDIA's apt
# repository (the image carries only Nsight Systems).
# KERNELS=name:launches a step,... replaces the list below (round19's launches a step; the encoder's kernels run
# once a layer a batch, about 3 a step).
# usage: ncu_run.sh <package> <runner dir> <units list> <n units> <steps to skip> <launches> [VAR=value ...]
set -uo pipefail
PKG=$1 RUNNER=$2 LIST=$3 N=$4 SKIP=$5 COUNT=$6; shift 6
B=${B:-/opt/wb}
UP=$S3/results/${AWS_BATCH_JOB_ID:-local}/logs/$(basename "$PWD")
for p in "$PKG" "$RUNNER"; do
  [ -d "$B/$p" ] || aws s3 cp --only-show-errors "$S3/$p.tgz" - | tar xz -C "$B"
done
apt-get update -qq > /dev/null 2>&1
# 2025.3: the CUDA 13.0 generation, as the hosts' driver 580 (a newer one may need a newer driver)
NCU_PKG=$(apt-cache search --names-only '^nsight-compute-2025\.3[0-9.]*$' | awk '{print $1}' | sort -V | tail -1)
apt-get install -y -qq --no-install-recommends "$NCU_PKG" > /dev/null 2>&1
NCU=$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V | tail -1)
echo "installed $NCU_PKG: $NCU"
export RUN_CACHE=$B/cache HF_HOME=$B/hf PYTHONPATH=$B/$PKG
export LD_LIBRARY_PATH=$B/venv/lib/python3.12/site-packages/nvidia/cublas/lib
KERNELS='cross_attention_kernel:32,Kernel2:134,kernel:182,tiled_split_gemm_kernel:29,residual_norm_kernel:102,reorder_append_parts_kernel:32,copy_parts_kernel:32,split_heads_bias_kernel:64,warp_softmax_forward:177,bias_add_vec_kernel:36,topk_stage_1:5,gemv2N_kernel:32,exact_attention_kernel:3,exact_attention_layout:3,regex:^(ampere|sm8|cutlass).*gemm:3'
for kv in "$@"; do export "$kv"; done
R=$PWD/root; mkdir -p "$R"; cp $B/$RUNNER/*.py "$R/"; cp $B/runner/units.json "$R/"; ln -sf $B/hf "$R/hf"
awk 'NF {print $1}' "$B/$LIST" | head -n "$N" > units.txt
$B/venv/bin/python -c "import json,sys; json.dump({'only_units': open(sys.argv[1]).read().split()}, open(sys.argv[2], 'w'))" \
  units.txt "$R/stop.json"
i=0
IFS=, read -ra LIST_K <<< "$KERNELS"
for KS in "${LIST_K[@]}"; do
  K=${KS%:*} KSKIP=$((${KS##*:} * SKIP))
  i=$((i + 1))
  export RUN_OUT=$PWD/out$i                                  # a fresh folder: the runner skips units already written
  mkdir -p "$RUN_OUT"
  t0=$(date +%s)
  (cd "$R" && timeout 1200 "$NCU" --target-processes all --kernel-name-base function -k "$K" \
    --launch-skip "$KSKIP" --launch-count "$COUNT" --kill yes \
    --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats \
    --section LaunchStats --section ComputeWorkloadAnalysis -f -o "$OLDPWD/report$i" \
    $B/venv/bin/python transcribe_run.py back "${MODE:-stream8}" > "$OLDPWD/ncu$i.out" 2>&1)
  echo "$K: exit $?, $(( $(date +%s) - t0 )) s, $(ls report$i.ncu-rep 2>/dev/null || echo no report)"
  if [ -f "report$i.ncu-rep" ]; then
    "$NCU" --import "report$i.ncu-rep" --page raw --csv > "report$i.csv" 2>/dev/null   # every metric collected
    aws s3 cp --only-show-errors "report$i.ncu-rep" "$UP/report$i.ncu-rep"
    aws s3 cp --only-show-errors "report$i.csv" "$UP/report$i.csv"
  else
    tail -5 "ncu$i.out"
  fi
done
ls -la report*
