#!/bin/bash
# One benchmark configuration on the box: N production runner processes started together, each in its own root
# with an only_units share of the list (every N-th unit), all reading the cache and writing out/<label>. Wall time
# from the first start to the last exit; the GPU (nvidia-smi) and CPU (vmstat) sampled throughout. MPS=1 runs them
# under NVIDIA MPS (kernels of several processes on the GPU at once); MODE=pipe16 etc. sets the runner's batch
# (default pipe8, production). Other VAR=value pairs go to the environment.
# usage: run.sh <label> <stock | package dir> <runner dir> <units list> <processes> [VAR=value ...]
set -euo pipefail
B=/opt/wb; LABEL=$1; PKG=$2; RUNNER=$3; LIST=$4; N=$5; shift 5
cd $B; OUT=$B/out/$LABEL; LOG=$B/logs/$LABEL
rm -rf "$OUT" "$LOG"; mkdir -p "$OUT" "$LOG"
export RUN_OUT=$OUT RUN_CACHE=$B/cache HF_HOME=$B/hf
export LD_LIBRARY_PATH=$B/venv/lib/python3.12/site-packages/nvidia/cublas/lib   # CT2 dlopens libcublas.so.12
[ "$PKG" = stock ] || export PYTHONPATH=$B/$PKG
for kv in "$@"; do export "$kv"; done
[ "${MPS:-0}" = 1 ] && nvidia-cuda-mps-control -d
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm \
  --format=csv,noheader,nounits -lms 500 > "$LOG/gpu.csv" & SMI=$!
vmstat -n 1 > "$LOG/cpu.txt" & VM=$!
t0=$(date +%s.%N); pids=()
for i in $(seq 0 $((N - 1))); do
  R=$LOG/root$i; mkdir -p "$R"; cp $B/$RUNNER/*.py "$R/"; cp $B/runner/units.json "$R/"; ln -s $B/hf "$R/hf"
  awk -v n="$N" -v i="$i" 'NF && (NR - 1) % n == i {print $1}' "$B/$LIST" > "$LOG/units.$i.txt"
  venv/bin/python -c "import json,sys; json.dump({'only_units': open(sys.argv[1]).read().split()}, open(sys.argv[2], 'w'))" \
    "$LOG/units.$i.txt" "$R/stop.json"
  (cd "$R" && exec $B/venv/bin/python transcribe_run.py back "${MODE:-pipe8}" > "$LOG/p$i.out" 2>&1) & pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
t1=$(date +%s.%N); kill $SMI $VM
[ "${MPS:-0}" = 1 ] && echo quit | nvidia-cuda-mps-control
venv/bin/python src/tools/turing/aws/box/summary.py "$LABEL" "$OUT" "$LOG" "$t0" "$t1" "$fail"
