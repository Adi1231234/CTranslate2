# Turing (sm_75) performance work

Goal: faster Whisper inference on an RTX 2080 (Windows, WDDM) with **bit-identical output**.
Every change is measured with `bench_whisper.py`: same clips, same decode parameters, and a hash of
every decoded token and score, which must match the stock `ctranslate2==4.8.2` wheel.

Kernel profile of stock 4.8.2 (`profile_kernels.py`, CUPTI activity records, large-v3, 8 clips):
- encoder: `cunn_SoftMaxForward` = 44.5% of GPU time. For 1500-wide rows `get_block_size`
  picks 1024 threads per row, so one row per SM and the kernel is latency-bound.
- decoder: cross-attention scores (`gemmSN_TN` via strided-batched GEMV) = 34.6%; every beam re-reads
  its clip's keys. ~1330 kernels per decode step.

Changes on branch `turing-perf`:
1. `src/ops/softmax_gpu.cu`: `warp_softmax_forward`, one warp per row for rows <= 2048. The max is
   exact in any order; the sum replays the legacy kernel's order (bit-exact).
   `CT2_CUDA_LEGACY_SOFTMAX=1` switches back to the legacy kernel for A/B runs.

Build notes (Windows): VS 2022 (MSVC 19.27 miscompiles pybind11 2.11), and OpenMP on
(`OPENMP_RUNTIME=COMP`): without OpenMP each worker thread owns a `thread_local BS::thread_pool`
whose destructor joins threads during thread exit, under the loader lock, so model teardown hangs.

Build: `tools/turing/build_windows.ps1` (CUDA 12.8, VS C++ tools, CMake, Ninja).
Measure: `python tools/turing/bench_whisper.py <sample_dir> <build_root>\pyct2`.
