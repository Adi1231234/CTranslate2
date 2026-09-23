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
1. `src/ops/softmax_kernels.cuh`: `warp_softmax_forward`, one warp per row for rows <= 2048. The max
   is exact in any order; the sum replays the legacy kernel's order. **Incident (fixed in 3cddbd2):**
   the first version summed only 2 terms per legacy thread, but a thread has 3 when cols = 2B + 1
   (65, 129, 257, 513, 1025). The encoder (1500) and the short-clip bench never hit those lengths;
   decoder self-attention does once a transcription passes ~61 tokens, so 382 production units were
   decoded differently and had to be redone with the stock wheel. `kernels/softmax_check.cu` now
   compares both kernels bit for bit on every row length 1..2048 (fp16/fp32, softmax/log-softmax,
   masked/unmasked) and, built against the buggy header (`run_probe.ps1 -Include`), flags exactly
   those 5 lengths.
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

Release gate (all four, every change): the kernel probes above; `bench_whisper.py` with
`BENCH_FIRST=60` (short), `90` and `118` (the longest clips, 200+ tokens); `prod_equiv.py` in the
production modes (`pipe8` and `exact2`, the fallback path) - every hash equal to the stock wheel's.

Results on the RTX 2080 at 724ae02 (softmax fixed + cross-attention scores), all hashes equal to stock:
- bench short / middle / long: 6a0ac320 / 25b0f78a / 67bd5eeb; encoder -23%, decoder about -20%.
- production engine on 150 real clips: pipe8 262ababd, 11.5x -> 14.0x realtime; exact2 a83ba880.
