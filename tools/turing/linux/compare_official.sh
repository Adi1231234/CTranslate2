#!/bin/bash
# Does a Linux build reproduce the official wheel's GPU kernels? Downloads the official ctranslate2 wheel
# (PyPI, manylinux, cp312) and compares the two libraries kernel by kernel with ../ptx_compare.py.
# usage: compare_official.sh <built package dir (…/ctranslate2)> [version]
set -euo pipefail
PKG=$(realpath "$1"); VER=${2:-4.8.2}
HERE=$(dirname "$(realpath "$0")"); WORK=/tmp/ct2-official-$VER
if [ ! -d "$WORK/wheel" ]; then
  mkdir -p "$WORK"
  python3 -m pip download -q "ctranslate2==$VER" --no-deps --only-binary=:all: --python-version 3.12 \
    --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 -d "$WORK"
  python3 -m zipfile -e "$WORK"/ctranslate2-*.whl "$WORK/wheel"
fi
OFFICIAL=$(find "$WORK/wheel" -name 'libctranslate2*.so*' | head -1)
BUILT=$(find "$PKG" -name 'libctranslate2.so.*' -type f | head -1)
echo "official $OFFICIAL"; echo "built    $BUILT"
python3 "$HERE/../ptx_compare.py" /usr/local/cuda/bin/cuobjdump "$OFFICIAL" "$BUILT"
