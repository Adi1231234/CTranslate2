#!/bin/bash
# The long-form sample (S3 data/longsample: lists and audio/) through ../../runner/longform_run.py (an entry.sh
# "script" line), once per run, with nvidia-smi sampling the GPU every second. Each run's rows (out/<label>) and logs go
# to the job's S3 results as soon as it ends, and a run is stopped after LONG_RUN_S seconds (default 1500), so a stuck
# run costs no more than that and loses nothing that finished (long1 hung to the job's limit).
# A run's settings may hold LONG_PROCS=<n> (n processes under MPS, every n-th recording each: LONG_SHARD) and
# NSYS=<delay s>:<seconds> (Nsight Systems on process 0 for that span; its kernel, NVTX and CUDA API summaries to S3).
# usage: longform.sh <runner dir> <label>:<package or stock>:<list file>:<VAR=value,...> ...
set -uo pipefail
RUNNER=$1; shift; B=${B:-/opt/wb}; RES=$S3/results/${AWS_BATCH_JOB_ID:-local}; HERE=$(pwd)
[ -d "$B/$RUNNER" ] || aws s3 cp --only-show-errors "$S3/$RUNNER.tgz" - | tar xz -C "$B"
aws s3 sync --only-show-errors "$S3/data/longsample/" "$B/longsample/"
echo "sample: $(ls $B/longsample/audio | wc -l) files, $(du -sh $B/longsample/audio | cut -f1)"
for run in "$@"; do
  IFS=: read -r label pkg list setting <<< "$run"
  [ "$pkg" = stock ] || [ -d "$B/$pkg" ] || aws s3 cp --only-show-errors "$S3/$pkg.tgz" - | tar xz -C "$B"
  procs=1; nsys=""
  for kv in ${setting//,/ }; do
    case $kv in LONG_PROCS=*) procs=${kv#LONG_PROCS=};; NSYS=*) nsys=${kv#NSYS=};; esac
  done
  echo "=== $(date +%T) $label $pkg $list ${setting//,/ }"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -lms 1000 \
    > "gpu_$label.csv" & SMI=$!
  [ "$procs" -gt 1 ] && nvidia-cuda-mps-control -d
  t0=$(date +%s.%N); pids=()
  for i in $(seq 0 $((procs - 1))); do
    wrap=""
    [ -n "$nsys" ] && [ "$i" = 0 ] && wrap="nsys profile -t cuda,nvtx --delay ${nsys%%:*} --duration ${nsys#*:} \
      -o $HERE/prof_$label --force-overwrite true"
    (export HF_HOME=$B/hf LONG_SHARD=$i/$procs; [ "$pkg" = stock ] || export PYTHONPATH=$B/$pkg
     for kv in ${setting//,/ }; do export "$kv"; done
     cd "$B/$RUNNER" && timeout "${LONG_RUN_S:-1500}" $wrap $B/venv/bin/python longform_run.py "$B/longsample/$list" \
       $B/longsample/audio "$B/out/$label/p$i" > "$HERE/log_${label}_$i.txt" 2>&1
     echo "exit $?" >> "$HERE/log_${label}_$i.txt") & pids+=($!)
  done
  wait "${pids[@]}"; t1=$(date +%s.%N)
  [ "$procs" -gt 1 ] && echo quit | nvidia-cuda-mps-control
  kill $SMI
  for i in $(seq 0 $((procs - 1))); do grep -E "STATS|RESULT|^exit|Error|error" "log_${label}_$i.txt" | tail -8; done
  cat log_${label}_*.txt | awk -v t0="$t0" -v t1="$t1" -v p="$procs" \
    '/^RESULT/ {for (i = 1; i <= NF; i++) if ($i ~ /^audio_h=/) {split($i, a, "="); h += a[2]}}
     END {w = t1 - t0; printf "TOTAL processes=%d audio_h=%.3f wall_s=%.1f x_realtime=%.2f\n", p, h, w, h * 3600 / w}'
  awk -F', ' '{u += $1; m = $2 > m ? $2 : m; w += $3; n++}
    END {if (n) printf "GPU: %.0f%% busy, peak %.0f MiB, %.0f W average, %d samples\n", u / n, m, w / n, n}' \
    "gpu_$label.csv"
  if [ -f "prof_$label.nsys-rep" ]; then
    nsys stats --report cuda_gpu_kern_sum,nvtx_sum,cuda_api_sum --format csv --output "prof_$label" "prof_$label.nsys-rep" \
      > /dev/null 2>&1
    # the GPU's time CUDA stream by stream (each worker has its own: the joint stream, the encoder, ladder lanes)
    aws s3 cp --only-show-errors "$S3/scripts/nsys_streams.py" . \
      && nsys export --type sqlite --output "prof_$label.sqlite" "prof_$label.nsys-rep" > /dev/null 2>&1 \
      && $B/venv/bin/python nsys_streams.py "prof_$label.sqlite" > "prof_${label}_streams.txt" 2>&1
    for f in prof_${label}*.csv prof_${label}_streams.txt prof_$label.nsys-rep; do aws s3 cp --only-show-errors "$f" "$RES/longform/$f"; done
  fi
  aws s3 sync --only-show-errors "$B/out/$label" "$RES/out/$label"
  for f in log_${label}_*.txt "gpu_$label.csv"; do aws s3 cp --only-show-errors "$f" "$RES/longform/$f"; done
done
