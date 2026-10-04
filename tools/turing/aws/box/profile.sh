#!/bin/bash
# Nsight Systems profile of the production runner: one process on the first N units of a list, CUDA and NVTX traced
# (no CPU sampling), exported to SQLite and summarized with ../../nsys_steps.py, nsys_streams.py and nsys_gaps.py.
# The report stays in logs/<label>/ (entry.sh uploads it).
# usage: profile.sh <label> <package dir> <runner dir> <units list> <n units> [VAR=value ...]
set -euo pipefail
B=/opt/wb; LABEL=$1; PKG=$2; RUNNER=$3; LIST=$4; N=$5; shift 5
LOG=$B/logs/$LABEL; OUT=$B/out/$LABEL; T=$B/src/tools/turing
rm -rf "$LOG" "$OUT"; mkdir -p "$LOG" "$OUT"
export RUN_OUT=$OUT RUN_CACHE=$B/cache HF_HOME=$B/hf PYTHONPATH=$B/$PKG
export LD_LIBRARY_PATH=$B/venv/lib/python3.12/site-packages/nvidia/cublas/lib
for kv in "$@"; do export "$kv"; done
R=$LOG/root0; mkdir -p "$R"; cp $B/$RUNNER/*.py "$R/"; cp $B/runner/units.json "$R/"; ln -s $B/hf "$R/hf"
awk 'NF {print $1}' "$B/$LIST" | head -n "$N" > "$LOG/units.txt"
$B/venv/bin/python -c "import json,sys; json.dump({'only_units': open(sys.argv[1]).read().split()}, open(sys.argv[2], 'w'))" \
  "$LOG/units.txt" "$R/stop.json"
t0=$(date +%s)
(cd "$R" && nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true -o "$LOG/report" \
  $B/venv/bin/python transcribe_run.py back "${MODE:-pipe8}" > "$LOG/nsys.out" 2>&1)
echo "profiled $(wc -l < "$LOG/units.txt") units in $(( $(date +%s) - t0 )) s: $(tail -2 "$R/progress.log" | head -1)"
nsys export --type=sqlite --force-overwrite=true -o "$LOG/report.sqlite" "$LOG/report.nsys-rep" > /dev/null
echo "== kernels by GPU time"
nsys stats --quiet --report cuda_gpu_kern_sum --format csv "$LOG/report.sqlite" 2>/dev/null | head -45 | tee "$LOG/kern_sum.csv"
for s in nsys_steps nsys_streams nsys_gaps; do
  echo "== $s"; $B/venv/bin/python "$T/$s.py" "$LOG/report.sqlite" 2>&1 | head -120 | tee "$LOG/$s.txt"
done
