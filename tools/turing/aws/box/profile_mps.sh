#!/bin/bash
# Nsight Systems on a production run of several MPS processes (an entry.sh "script" line): run.sh under nsys,
# which follows the runner processes, CUDA and NVTX traced over a steady window (--delay after the start); MPS
# runs outside the trace. The report is exported to SQLite in this folder (entry.sh uploads it).
# usage: profile_mps.sh <label> <package> <runner dir> <units list> <processes> <delay s> <window s> [VAR=value ...]
set -uo pipefail
LABEL=$1 PKG=$2 RUNNER=$3 LIST=$4 N=$5 DELAY=$6 WIN=$7; shift 7
B=${B:-/opt/wb}
for p in "$PKG" "$RUNNER"; do
  [ -d "$B/$p" ] || aws s3 cp --only-show-errors "$S3/$p.tgz" - | tar xz -C "$B"
done
nvidia-cuda-mps-control -d
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --delay="$DELAY" --duration="$WIN" \
  --force-overwrite=true -o report \
  bash "$B/src/tools/turing/aws/box/run.sh" "$LABEL" "$PKG" "$RUNNER" "$LIST" "$N" "$@" > nsys.out 2>&1
echo quit | nvidia-cuda-mps-control
tail -4 nsys.out
nsys export --type=sqlite --force-overwrite=true -o report.sqlite report.nsys-rep > /dev/null 2>&1
ls -la report.sqlite
