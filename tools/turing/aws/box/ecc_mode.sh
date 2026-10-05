#!/bin/bash
# Sets the GPU's ECC mode for the rest of a job (an entry.sh "script" line, privileged job): 0 off, 1 on, applied
# by a GPU reset, which needs no process on the GPU (run.sh stops MPS after each configuration). The L40S keeps its
# GDDR6 ECC inline: 46068 of 49140 MiB with it on, its check bits read with the data, and the cross-attention moved
# 225 MB from DRAM a launch for 192 MB of keys and values (round27 ncu). A job that sets 0 sets 1 again at its end:
# Batch may give the instance to the next job.
# usage: ecc_mode.sh <0|1>
set -uo pipefail
query() { nvidia-smi --query-gpu=ecc.mode.current,ecc.mode.pending,memory.total --format=csv,noheader; }
echo "before: $(query)"
nvidia-smi -e "$1"
nvidia-smi --gpu-reset -i 0 || echo "the reset failed: the mode stays pending"
echo "after: $(query)"
