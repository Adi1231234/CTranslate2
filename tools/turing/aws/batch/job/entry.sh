#!/bin/bash
# The whisper-bench job (the image's entrypoint): units.json and the cached units from S3, then every line of
# $EXPERIMENTS in order, then results and logs to s3://$BUCKET/$S3_PREFIX/results/<job id>/. A line is either
#   label|package|runner dir|units list|processes|VAR=value ...   one configuration (../../box/run.sh)
#   compare|reference label|label                                  compare.py on the two outputs
#   profile<name>|package|runner dir|units list|n units|VAR=value   an Nsight Systems profile (../../box/profile.sh)
#   script<name>|<key under $S3_PREFIX>|arg ...                    a bash script from S3, run in logs/<label>/
#                                                                  (e.g. ../../box/probes.sh: kernel probes)
set -uo pipefail
B=/opt/wb; S3=s3://$BUCKET/$S3_PREFIX; OUT=$S3/results/${AWS_BATCH_JOB_ID:-local}
cd $B
aws s3 cp --only-show-errors "$S3/data/units.json" runner/units.json && cp runner/units.json runner_old/
aws s3 sync --only-show-errors "$S3/data/cache/" cache/
imds() { curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" "http://169.254.169.254/latest/meta-data/$1"; }
echo "host $(imds instance-type) $(imds placement/availability-zone), cached units $(ls cache | wc -l)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "vCPU $(nproc), memory $(free -g | awk '/Mem:/ {print $2}') GB, MPS control: $(command -v nvidia-cuda-mps-control || echo none)"
: > results.jsonl
while IFS='|' read -r label a b c d extra; do
  [ -z "$label" ] && continue
  extra=${extra//|/ }                                   # read leaves the later fields' separators in the last one
  echo "=== $(date +%T) $label $a $b $extra"
  # A package or runner the image does not carry comes from S3 (packed as <name>.tgz with a top folder <name>:
  # ../../linux/build.sh's output, or a runner folder).
  if [[ "$label" != compare && "$label" != script* ]]; then
    for p in "$a" "$b"; do
      [ -d "$p" ] || aws s3 cp --only-show-errors "$S3/$p.tgz" - | tar xz -C $B
    done
    cat "$a/ctranslate2/BUILD.txt"
  fi
  if [ "$label" = compare ]; then
    venv/bin/python src/tools/turing/scale/compare.py "out/$a" "out/$b" | tee "logs/compare_${a}_${b}.txt"
  elif [[ "$label" == profile* ]]; then
    bash src/tools/turing/aws/box/profile.sh "$label" "$a" "$b" "$c" "$d" $extra
  elif [[ "$label" == script* ]]; then
    mkdir -p "logs/$label" && aws s3 cp --only-show-errors "$S3/$a" "logs/$label/run.sh"
    (cd "logs/$label" && B=$B S3=$S3 LD_LIBRARY_PATH=$B/venv/lib/python3.12/site-packages/nvidia/cublas/lib \
      bash run.sh $b $c $d $extra 2>&1 | tee out.txt)
  else
    bash src/tools/turing/aws/box/run.sh "$label" "$a" "$b" "$c" "$d" $extra
    tail -2 "logs/$label/root0/progress.log"
  fi
  aws s3 cp --only-show-errors results.jsonl "$OUT/results.jsonl"
done <<< "$EXPERIMENTS"
aws s3 sync --only-show-errors --no-follow-symlinks logs/ "$OUT/logs/" --exclude "*.py"   # roots link the model
aws s3 sync --only-show-errors out/ "$OUT/out/"
echo "=== $(date +%T) done"
