#!/bin/bash
# Kernels and the GPU's hardware counters in one Nsight Systems report (an entry.sh "script" line; the job must be
# privileged, see ../batch/submit.py): run.sh under nsys with CUDA traced and DRAM bandwidth etc. sampled every
# 100 us over a steady window, MPS outside the trace when MPS=1 is given. ../../nsys_dram.py splits the DRAM
# traffic between kernel kinds from the report exported to SQLite here (entry.sh uploads it).
# usage: profile_metrics.sh <label> <package> <runner dir> <units list> <processes> <delay s> <window s> [VAR=value]
set -uo pipefail
LABEL=$1 PKG=$2 RUNNER=$3 LIST=$4 N=$5 DELAY=$6 WIN=$7; shift 7
B=${B:-/opt/wb}
for p in "$PKG" "$RUNNER"; do
  [ -d "$B/$p" ] || aws s3 cp --only-show-errors "$S3/$p.tgz" - | tar xz -C "$B"
done
mps=0; args=()
for kv in "$@"; do [ "$kv" = MPS=1 ] && mps=1 || args+=("$kv"); done
[ $mps = 1 ] && nvidia-cuda-mps-control -d
nsys profile --trace=cuda --sample=none --cpuctxsw=none --gpu-metrics-devices=all --gpu-metrics-frequency=10000 \
  --delay="$DELAY" --duration="$WIN" --force-overwrite=true -o report \
  bash "$B/src/tools/turing/aws/box/run.sh" "$LABEL" "$PKG" "$RUNNER" "$LIST" "$N" "${args[@]}" > nsys.out 2>&1
[ $mps = 1 ] && echo quit | nvidia-cuda-mps-control
tail -4 nsys.out
nsys export --type=sqlite --force-overwrite=true -o report.sqlite report.nsys-rep > /dev/null 2>&1
ls -la report.sqlite
