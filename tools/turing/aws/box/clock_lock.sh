#!/bin/bash
# Locks the GPU's SM clock to a range for the rest of a job (an entry.sh "script" line, privileged job), or
# releases the lock (no argument). The batched path sits at the 350 W limit with GPU Boost moving the SM clock
# between ~1500 and 2520 MHz as the kernels change; a steady clock may spend less energy for the same work (power
# grows faster than the clock), and a memory-bound kernel loses nothing at a lower one.
# usage: clock_lock.sh [min MHz] [max MHz]
set -uo pipefail
if [ $# -ge 2 ]; then
  nvidia-smi -lgc "$1,$2"
else
  nvidia-smi -rgc
fi
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.limit --format=csv
