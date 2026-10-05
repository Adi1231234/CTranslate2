#!/bin/bash
# The GPU's power limits (an entry.sh "script" line, privileged job): prints the current, default, minimum and
# maximum board power limits and, with an argument, sets the limit to that many watts for the rest of the job
# (the job's later lines run under it; the host resets it when the instance goes).
# usage: power_limit.sh [watts]
set -uo pipefail
nvidia-smi --query-gpu=name,power.limit,power.default_limit,power.min_limit,power.max_limit,clocks.max.sm \
  --format=csv
if [ $# -ge 1 ]; then
  nvidia-smi -pl "$1"
  nvidia-smi --query-gpu=power.limit --format=csv,noheader
fi
