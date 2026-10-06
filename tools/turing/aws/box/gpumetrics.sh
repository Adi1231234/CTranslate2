#!/bin/bash
# Device-wide hardware counters while a production run goes (an entry.sh "script" line): run.sh starts in the
# background, and after a warm-up Nsight Systems samples the GPU's metrics (DRAM bandwidth, SM and tensor-core
# activity; all processes, MPS included) for a fixed window; the table is summarized as averages and percentiles.
# usage: gpumetrics.sh <label> <package> <runner dir> <units list> <processes> <warm-up s> <window s> [VAR=value ...]
set -uo pipefail
LABEL=$1 PKG=$2 RUNNER=$3 LIST=$4 N=$5 WARM=$6 WIN=$7; shift 7
B=${B:-/opt/wb}
[ -d "$B/$PKG" ] || { aws s3 cp --only-show-errors "$S3/$PKG.tgz" - | tar xz -C "$B"; }
bash "$B/src/tools/turing/aws/box/run.sh" "$LABEL" "$PKG" "$RUNNER" "$LIST" "$N" "$@" > run.out 2>&1 &
RUN=$!
sleep "$WARM"
nsys profile --trace=none --sample=none --cpuctxsw=none --gpu-metrics-devices=all --gpu-metrics-frequency=2000 \
  --duration="$WIN" --force-overwrite=true -o metrics sleep "$((WIN + 5))" > nsys.out 2>&1
tail -3 nsys.out
nsys export --type=sqlite --force-overwrite=true -o metrics.sqlite metrics.nsys-rep > /dev/null 2>&1
aws s3 cp --only-show-errors "$S3/scripts/gpu_metrics_summary.py" . && $B/venv/bin/python gpu_metrics_summary.py metrics.sqlite
wait $RUN
tail -2 run.out
