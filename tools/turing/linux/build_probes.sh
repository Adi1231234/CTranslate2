#!/bin/bash
# Build kernel probes (../kernels/*.cu) on Linux as ../kernels/build_probe.ps1 does on Windows: nvcc against the
# library's headers and thrust/cub, linked to cuBLAS. Default code: sm_86 SASS, what the 8.6+PTX library runs on
# sm_86 and sm_89 GPUs (AWS A10G, L4, L40S). Output: <out>/probes.tgz with one executable per probe; at run time
# put the production venv's nvidia/cublas/lib on LD_LIBRARY_PATH (the cuBLAS build the replicas are checked on).
# usage: build_probes.sh <source checkout> <out dir> <probe> [probe ...]     (after setup_env.sh)
set -euo pipefail
SRC=$(realpath "$1"); OUT=$2; shift 2
GENCODE=${GENCODE:--gencode=arch=compute_86,code=sm_86}
CCCL="$SRC/third_party/thrust"
BIN="$OUT/probes"; rm -rf "$BIN"; mkdir -p "$BIN"
for name in "$@"; do
  /usr/local/cuda/bin/nvcc -O3 -std=c++17 $GENCODE --expt-relaxed-constexpr -diag-suppress 2219 \
    -I "$CCCL/cub" -I "$CCCL/thrust" -I "$CCCL/libcudacxx/include" -I "$SRC/src" -I "$SRC/include" \
    -o "$BIN/$name" "$SRC/tools/turing/kernels/$name.cu" -lcublas -lcublasLt
  echo "built $name"
done
git -C "$SRC" log --oneline -1 > "$BIN/BUILD.txt"
tar czf "$OUT/probes.tgz" -C "$OUT" probes
echo "$OUT/probes.tgz: $*"
