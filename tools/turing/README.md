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
   decoded differently and have to be redone with a build that passes the gate.
   `kernels/softmax_check.cu` now compares both kernels bit for bit on every row length 1..2048
   (fp16/fp32, softmax/log-softmax, masked/unmasked) and, built against the buggy header
   (`run_probe.ps1 -Include`), flags exactly those 5 lengths.
2. `src/cuda/attention_scores_k64.cuh`: cross-attention scores (k = 64, n = 1500, 1-8 queries) with
   the exact fp32 arithmetic of `gemmSN_TN`, one thread per key. The order was recovered with
   `kernels/qk_probe.cu` (cancellation triples give the leaf set of every node of the summation
   tree): per output, 16 partials p_r = k[r]q[r] + k[r+16]q[r+16] + k[r+32]q[r+32] + k[r+48]q[r+48]
   left to right, then +0 + p_0 + ... + p_15, then half_rn(alpha * sum). `kernels/qk_check.cu`
   compares it with the cuBLAS call bit for bit: 0 mismatches over 2.8e10 outputs for batch
   2..1024 and 1..8 queries. For batch 1 cuBLAS picks another kernel, so the route is limited to the
   verified shapes, to sm_75 and to cuBLAS 12.9.2 (`cublas_replicas_verified`).
3. DisableTokens `add_range` (landed in 2a3cac8): Whisper's timestamp rules disable whole token
   ranges; they are filled as ranges instead of listing (and uploading) every index.
4. `src/cuda/timestamp_rules.cu` (b647c23): the Whisper timestamp rule reads its cub reductions back
   once per decode step instead of three host syncs per row. `kernels/ts_check.cu` checks it.
5. `src/cuda/disable_tokens.cu` (e6d9e09): DisableTokens on the GPU takes its ranges, all-row ids and
   single indices as kernel arguments (no host-to-device copy, no stream sync). No measurable gain
   on its own; kept.
6. `src/cuda/allocator.cc` (71841fb): `cuda_malloc_async` draws from the device's default memory pool,
   whose release threshold is 0 by default, so every synchronize returned the freed memory to the OS
   and the next allocations mapped it again: 25-33 ms GPU idle gaps (`nsys_biggaps.py`). The pool now
   keeps it (threshold UINT64_MAX per device at its first allocation, as PyTorch's cudaMallocAsync
   backend); `clear_cache()` trims the pool, so unloading a model still frees it;
   `CT2_CUDA_ASYNC_ALLOCATOR_RELEASE_THRESHOLD` overrides (docs/environment_variables.md). Pool peak
   in production is about 6.4 GB of 8 GB.
Production setting, not a code change: `cpu_threads=1`. The OpenMP threads of the default only spin.
Ruled out: the batched pipeline's up-front log-mel extraction is 0.53 s of 42 s (`feature_time.py`).

Build notes (Windows): VS 2022 (MSVC 19.27 miscompiles pybind11 2.11), and OpenMP on
(`OPENMP_RUNTIME=COMP`): without OpenMP each worker thread owns a `thread_local BS::thread_pool`
whose destructor joins threads during thread exit, under the loader lock, so model teardown hangs.

Build: `build_windows.ps1` (CUDA 12.8, VS C++ tools, CMake, Ninja; output in a staging package dir).
Measure: `python tools/turing/bench_whisper.py <sample_dir> <package parent dir>`.
Probes: `kernels/run_probe.ps1 <softmax_check|qk_check|ts_check|qk_probe|qk_diff>` (nvcc sm_75,
production's cuBLAS DLL). Profiles: Nsight Systems + `nsys_gaps.py`, `nsys_cpu.py`, `nsys_biggaps.py`.

Verification. Every change must leave the output byte-identical to the stock wheel's:
- Inner loop, every change: `digest.py` (`host/digest.ps1 -Pkgs <build>`), one process, about 40 s
  with the model load. It hashes every byte the model returns (encoder outputs at batch 8/8/6/4/1;
  beam 5 with all 5 hypotheses, scores and no-speech probability, natural ends and 448 forced steps
  = every self-attention width; batch 1 after a previous-text prompt; the seeded fallback sampling
  with every step's full-vocabulary logits) and compares each section with a golden file recorded
  ONCE from the stock wheel (`-Record`). The golden file belongs to one GPU + cuBLAS build: record
  one per machine. Validated on the RTX 2080 (b6919d8): stock vs its golden PASS, 71841fb PASS, the
  buggy softmax build (9bb9577) FAIL in 6 of 12 sections.
- Before a production deploy: the full gate, `host/gate.ps1` (about 4.5 min): the kernel probes,
  `bench_whisper.py` with `BENCH_FIRST=60` (short), `90` and `118` (the longest clips, 200+ tokens),
  and `prod_equiv.py` in both production modes (`pipe8` and `exact2`, the fallback path). Ends with
  `GATE PASS` or `GATE FAIL` in `D:\ct2build\ab.log`.
- Kernel probes (`softmax_check`, `qk_check`, `ts_check`) when kernel sources change. `qk_check`
  counts only the routed shapes in its TOTAL; the batch 1 control is printed on its own line.
- Speed: a separate A/B, `host/equiv_ab.ps1` (prod_equiv pipe8 on 150 clips, configurations
  alternated, nvidia-smi samples). Discard the first run after idle: it is slower.

Host scripts (`host/`, for the Yarin layout; start long ones detached, log in `D:\ct2build\ab.log`):
`prod.ps1` shared helpers (pause/resume production, timed python runs); `gate.ps1` the full gate;
`digest.ps1` the fast check; `equiv_ab.ps1` the speed A/B; `ab.ps1` builds and A/Bs with production
paused; `run_paused.ps1` runs one tool paused; `deploy.ps1` swaps `pyct2-next` into `pyct2` and ends
with Resume-Production, which STARTS the production supervisor.

Results on the RTX 2080, production engine (`prod_equiv.py` pipe8, 150 real clips), every hash equal
to stock (pipe8 262ababd, exact2 a83ba880; bench 6a0ac320 / 25b0f78a / 67bd5eeb):
- stock 11.5x realtime (13.1x with `cpu_threads=1`)
- softmax + cross-attention scores (724ae02) 14.0x, `cpu_threads=1` 15.5x, `add_range` 17.7x,
  timestamp rule 19.3x, DisableTokens on the GPU 19.3x, allocator (71841fb) 20.0x (40.6 s).
