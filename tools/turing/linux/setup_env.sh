#!/bin/bash
# Build environment for this fork on Ubuntu 24.04 (a throwaway WSL distro or a cloud box), run as root.
# CUDA comes from NVIDIA's apt repository at the versions the official Linux wheel is built with
# (python/tools/prepare_build_environment_linux.sh): nvcc 12.8.93 decides the GPU code, so it is pinned.
# The CPU backends (MKL, oneDNN) and tensor parallelism are left out: this build is for GPU use only.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends build-essential cmake ninja-build git ca-certificates curl \
  python3-dev python3-venv patchelf
curl -fsSL -o /tmp/cuda-keyring.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb >/dev/null
apt-get update -qq

pinned() {  # the repository's exact version string for a package at an upstream version, or fail
  local v
  v=$(apt-cache madison "$1" | awk '{print $3}' | grep "^$2" | head -1)
  [ -n "$v" ] || { echo "no $1 $2 in the NVIDIA repository" >&2; exit 1; }
  echo "$1=$v"
}
apt-get install -y -qq --no-install-recommends \
  "$(pinned cuda-nvcc-12-8 12.8.93)" "$(pinned cuda-cudart-dev-12-8 12.8.90)" \
  "$(pinned libcublas-dev-12-8 12.8.4.1)" "$(pinned libcurand-dev-12-8 10.3.9.90)" cuda-cuobjdump-12-8
ln -sfn /usr/local/cuda-12.8 /usr/local/cuda
/usr/local/cuda/bin/nvcc --version | tail -2
gcc --version | head -1
cmake --version | head -1
