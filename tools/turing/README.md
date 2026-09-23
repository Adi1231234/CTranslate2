# Turing (sm_75) performance work

Goal: faster Whisper inference on an RTX 2080 (Windows, WDDM) with **bit-identical output**.
Every change is measured with `bench_whisper.py`: same clips, same decode parameters, and hashes of
every encoder output byte (`enc_sha`), every token + full-precision score + no-speech probability
(`full_sha`), which must match the stock `ctranslate2==4.8.2` wheel.

Kernel profile of stock 4.8.2 (`profile_kernels.py`, `trace_kernels.py` + `trace_summary.py`; CUPTI
activity records, large-v3, 8 clips, beam 5):
- encoder: `cunn_SoftMaxForward` = 44.5% of GPU time. For 1500-wide rows `get_block_size` picks
  1024 threads per row, so one row per SM and the kernel is latency-bound.
- decoder: cross-attention scores = 40% of a decode step. CTranslate2 already shares the keys
  across beams (queries reshaped to [batch, heads, beams, 64]); the cost is cuBLAS's `gemmSN_TN`
  kernel, which spreads the 1500 keys over blocks of 8 and reads them at ~85 GB/s.

Changes on branch `turing-perf` (`CT2_CUDA_STOCK_KERNELS=1` restores every upstream kernel):
1. `src/ops/softmax_gpu.cu`: `warp_softmax_forward`, one warp per row for rows <= 2048. The max is
   exact in any order; the sum replays the legacy kernel's order.
2. `src/cuda/attention_scores_k64.cuh`: cross-attention scores (k = 64, n = 1500, 1-8 queries) with
   the exact fp32 arithmetic of `gemmSN_TN`, one thread per key. The order was recovered with
   `kernels/qk_probe.cu` (cancellation triples give the leaf set of every node of the summation
   tree): per output, 16 partials p_r = k[r]q[r] + k[r+16]q[r+16] + k[r+32]q[r+32] + k[r+48]q[r+48]
   left to right, then +0 + p_0 + ... + p_15, then half_rn(alpha * sum). `kernels/qk_check.cu`
   compares it with the cuBLAS call bit for bit: 0 mismatches over 2.8e10 outputs for batch
   2..1024 and 1..8 queries. For batch 1 cuBLAS picks another kernel, so the route is limited to the
   verified shapes, to sm_75 and to cuBLAS 12.9.2 (`cublas_replicas_verified`).

Build notes (Windows): VS 2022 (MSVC 19.27 miscompiles pybind11 2.11), and OpenMP on
(`OPENMP_RUNTIME=COMP`): without OpenMP each worker thread owns a `thread_local BS::thread_pool`
whose destructor joins threads during thread exit, under the loader lock, so model teardown hangs.

Build: `build_windows.ps1` (CUDA 12.8, VS C++ tools, CMake, Ninja; output in a staging package dir).
Measure: `python tools/turing/bench_whisper.py <sample_dir> <package parent dir>`.
Probes: `kernels/run_probe.ps1 <qk_probe|qk_check|qk_diff>` (nvcc sm_75, production's cuBLAS DLL).
Host scripts (`host/`): `ab.ps1` builds and A/Bs with the production run paused, `run_paused.ps1`
runs one tool paused, `deploy.ps1` swaps the package.

Results on the RTX 2080 (bench_whisper.py, 4 x 8 clips, large-v3, beam 5), all hashes identical:
- stock 4.8.2: E 5.03-5.13 s, D 5.67-5.93 s.
- softmax: E 3.94-4.03 s (-21%).
- softmax + cross-attention scores: D 4.67-5.01 s (-15% to -19%).
- production (crowd-transcribe-v5, batch 8 + fallback): 11x -> 13.1x realtime with the softmax.
