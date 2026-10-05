#!/bin/bash
# The long-form sample (S3 data/longsample: lists and audio/) through ../../runner/longform_run.py (an entry.sh
# "script" line), once per run, each a fresh process with nvidia-smi sampling the GPU every second; rows to
# out/<label> (synced to S3 with the job's other outputs).
# usage: longform.sh <runner dir> <label>:<package or stock>:<list file>:<VAR=value,...> ...
set -uo pipefail
RUNNER=$1; shift; B=${B:-/opt/wb}
[ -d "$B/$RUNNER" ] || aws s3 cp --only-show-errors "$S3/$RUNNER.tgz" - | tar xz -C "$B"
aws s3 sync --only-show-errors "$S3/data/longsample/" "$B/longsample/"
echo "sample: $(ls $B/longsample/audio | wc -l) files, $(du -sh $B/longsample/audio | cut -f1)"
for run in "$@"; do
  IFS=: read -r label pkg list setting <<< "$run"
  [ "$pkg" = stock ] || [ -d "$B/$pkg" ] || aws s3 cp --only-show-errors "$S3/$pkg.tgz" - | tar xz -C "$B"
  echo "=== $(date +%T) $label $pkg $list ${setting//,/ }"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -lms 1000 \
    > "gpu_$label.csv" & SMI=$!
  (export HF_HOME=$B/hf; [ "$pkg" = stock ] || export PYTHONPATH=$B/$pkg
   for kv in ${setting//,/ }; do export "$kv"; done
   cd "$B/$RUNNER" && $B/venv/bin/python longform_run.py "$B/longsample/$list" $B/longsample/audio "$B/out/$label" \
     2>&1 | grep -vE "[Ww]arning|warn\(" | tail -70)
  kill $SMI
  awk -F', ' '{u += $1; m = $2 > m ? $2 : m; w += $3; n++}
    END {if (n) printf "GPU: %.0f%% busy, peak %.0f MiB, %.0f W average, %d samples\n", u / n, m, w / n, n}' \
    "gpu_$label.csv"
done
