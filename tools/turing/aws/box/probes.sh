#!/bin/bash
# Kernel probes on the job's GPU (an entry.sh "script" line runs this from S3): unpacks the probes.tgz that
# ../../linux/build_probes.sh made and runs each probe given as name[:arg,arg...], in order, against the
# production venv's cuBLAS (entry.sh sets LD_LIBRARY_PATH).
# usage: probes.sh <key of probes.tgz under $S3> <name[:args]> ...
set -uo pipefail
aws s3 cp --only-show-errors "$S3/$1" probes.tgz && tar xzf probes.tgz && shift
echo "probes from $(cat probes/BUILD.txt) on $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
ls "$LD_LIBRARY_PATH"
for p in "$@"; do
  name=${p%%:*}; args=""; [[ "$p" == *:* ]] && args=${p#*:}
  echo "=== $(date +%T) $name ${args//,/ }"
  timeout 2400 "./probes/$name" ${args//,/ }
  echo "=== exit $?"
done
