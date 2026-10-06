#!/bin/bash
# A kernel probe under Nsight Systems (an entry.sh "script" line): its CUDA kernels and NVTX ranges, as CSVs next to
# this script's output (synced to S3 with the job's logs), e.g. which cuBLAS kernel a shape runs
# (../../kernels/selfattn_kernels.cu).
# usage: nsys_probe.sh <key of probes.tgz under $S3> <probe> [args]
set -uo pipefail
aws s3 cp --only-show-errors "$S3/$1" probes.tgz && tar xzf probes.tgz && shift
name=$1; shift
echo "probes from $(cat probes/BUILD.txt) on $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
nsys profile -t cuda,nvtx -o "$name" --force-overwrite true "./probes/$name" "$@" 2>&1 | tail -3
nsys stats --report cuda_gpu_trace,nvtx_gpu_proj_trace --format csv --output "$name" "$name.nsys-rep" > /dev/null 2>&1
ls -la ${name}*.csv
