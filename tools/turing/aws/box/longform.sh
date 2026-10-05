#!/bin/bash
# The long-form sample (S3 data/longsample: lists and audio/) through ../../runner/longform_run.py (an entry.sh
# "script" line), once per run, each a fresh process with nvidia-smi sampling the GPU every second. Each run's rows
# (out/<label>) and log go to the job's S3 results as soon as it ends, and a run is stopped after LONG_RUN_S seconds
# (default 1500), so a stuck run costs no more than that and loses nothing that finished (long1 hung to the job's limit
# and the job's end-of-job sync never came).
# usage: longform.sh <runner dir> <label>:<package or stock>:<list file>:<VAR=value,...> ...
set -uo pipefail
RUNNER=$1; shift; B=${B:-/opt/wb}; RES=$S3/results/${AWS_BATCH_JOB_ID:-local}
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
   cd "$B/$RUNNER" && timeout "${LONG_RUN_S:-1500}" $B/venv/bin/python longform_run.py "$B/longsample/$list" \
     $B/longsample/audio "$B/out/$label" > "$OLDPWD/log_$label.txt" 2>&1; echo "exit $?" >> "$OLDPWD/log_$label.txt")
  kill $SMI
  grep -vE "[Ww]arning|warn\(" "log_$label.txt" | tail -70
  awk -F', ' '{u += $1; m = $2 > m ? $2 : m; w += $3; n++}
    END {if (n) printf "GPU: %.0f%% busy, peak %.0f MiB, %.0f W average, %d samples\n", u / n, m, w / n, n}' \
    "gpu_$label.csv"
  aws s3 sync --only-show-errors "$B/out/$label" "$RES/out/$label"
  aws s3 cp --only-show-errors "log_$label.txt" "$RES/longform/log_$label.txt"
  aws s3 cp --only-show-errors "gpu_$label.csv" "$RES/longform/gpu_$label.csv"
done
