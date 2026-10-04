#!/bin/bash
# Build this fork on Linux for one GPU arch, with the official wheel's CUDA flags minus the CPU backends,
# as ../build_windows.ps1 does on Windows. The default arch 8.6+PTX matches what the official wheel runs on
# sm_86 and sm_89 GPUs (AWS A10G, L4, L40S): its "Common" list stops at sm_86 SASS plus compute_86 PTX, and
# such a GPU runs the sm_86 SASS. Output: <out>/ctranslate2, a drop-in package (put <out> first on sys.path);
# the library sits next to the extension, which finds it through $ORIGIN.
# usage: build.sh <source checkout> <out dir> [arch]      (after setup_env.sh)
set -euo pipefail
SRC=$(realpath "$1"); OUT=$2; ARCH=${3:-8.6+PTX}
BUILD="$SRC/build-linux-sm${ARCH//[.+]/}"; INST="$BUILD/install"
if [ ! -f "$BUILD/build.ninja" ]; then
  cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INST" \
    -DCMAKE_CXX_FLAGS="-msse4.1" -DBUILD_CLI=OFF -DWITH_MKL=OFF -DOPENMP_RUNTIME=COMP \
    -DWITH_CUDA=ON -DWITH_CUDNN=OFF -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DCUDA_DYNAMIC_LOADING=ON \
    -DCUDA_NVCC_FLAGS="-Xfatbin=-compress-all" -DCUDA_ARCH_LIST="$ARCH"
fi
cmake --build "$BUILD" --target install --parallel

# The Python extension against that library, with the pybind11 the Windows builds use.
VENV="$BUILD/pyvenv"
[ -x "$VENV/bin/python" ] || { python3 -m venv "$VENV"; "$VENV/bin/pip" install -q pybind11==2.11.1 setuptools wheel; }
(cd "$SRC/python" && CTRANSLATE2_ROOT="$INST" "$VENV/bin/python" setup.py build_ext --inplace)

PKG="$OUT/ctranslate2"
rm -rf "$PKG"; mkdir -p "$OUT"
cp -r "$SRC/python/ctranslate2" "$PKG"
cp -P "$INST"/lib/libctranslate2.so* "$PKG/"
# The OpenMP runtime ships inside the package, as the official wheel does (libiomp5 in ctranslate2.libs) and
# ../build_windows.ps1 does with vcomp140.dll: a slim image has no libgomp. RUNPATH is not inherited, so the
# library gets $ORIGIN too.
cp -L "$(gcc -print-file-name=libgomp.so.1)" "$PKG/"
patchelf --set-rpath '$ORIGIN' "$PKG"/_ext*.so "$PKG"/libctranslate2.so.*.*.*
git -C "$SRC" log --oneline -1 > "$PKG/BUILD.txt"
echo "built $PKG from $(git -C "$SRC" rev-parse --short HEAD) for $ARCH"
