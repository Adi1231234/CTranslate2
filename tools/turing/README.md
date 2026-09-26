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
7. `src/ops/softmax_rows1024.cuh` (514b95f): fp16 rows of 1026..2048 without lengths (the encoder's
   1500). Each lane keeps its 32 legacy threads' values in registers (no shared memory, so no cap
   of 8 warps per SM); same float operations as warp_softmax_forward. 8.19 -> 3.99 ms per 8-clip
   layer, 81% of bandwidth (`kernels/softmax_bench.cu`).
8. `src/cuda/split_heads.cu`, `src/layers/split_heads_fused.cc` (98a51b4): Dense bias add, head split
   and Q/K/V split in one kernel (encoder and decoder QKV, cross-attention queries and memory K/V,
   `Dense::compute_without_bias`). The add is `__hadd`, which is `cuda::plus<__half>` on sm_53+.
9. Deferred beam reorder (6711bf1, c811fd7): `Decoder::update_state` leaves the beam order in
   `state["pending_beam_reorder"]` and the next step's self-attention reorders and appends its keys
   and values in one launch (`cuda/cache_reorder.cu`, `layers/kv_cache.cc`) instead of a Gather and
   a Concat of every cache; flushed before a beam search returns.
10. `src/ops/bias_add_vec.cuh` (0a8b952): fp16 BiasAdd on 16-byte vectors (plain, residual, GELU),
   same `__hadd` / `gelu_func` arithmetic as the thrust paths.
11. Timestamp rule (5650901): all rows' maxima and exp sums in two `cub::DeviceSegmentedReduce`
   launches instead of four kernels per row; in this CCCL SegmentedReducePolicy is SingleTilePolicy,
   so the sums keep their order (`kernels/ts_check.cu`).
12. Stream priorities (e1a636e): worker streams at the highest priority, Whisper's encoder on the
   thread's low-priority stream (`cuda::UseLowPriorityStreamInScope`). Without it the pipelined
   encoder's big kernels delayed every decoder kernel (batch8 and pipe8 took the same time); now
   the encoder fills the decoder's gaps. Only the order in which the GPU takes up kernels changes.
Production setting, not a code change: `cpu_threads=1`. The OpenMP threads of the default only spin.
Ruled out: the batched pipeline's up-front log-mel extraction is 0.53 s of 42 s (`feature_time.py`).

Build notes (Windows): VS 2022 (MSVC 19.27 miscompiles pybind11 2.11), and OpenMP on
(`OPENMP_RUNTIME=COMP`): without OpenMP each worker thread owns a `thread_local BS::thread_pool`
whose destructor joins threads during thread exit, under the loader lock, so model teardown hangs.

Build: `build_windows.ps1` (CUDA 12.8, VS C++ tools, CMake, Ninja; output in a staging package dir).
Without a CUDA install: `fetch_cuda.ps1 -Dest <dir>` unpacks the official build's CUDA 12.8.1 components
from NVIDIA's redistributable archives (sha256-checked); set `CUDA_PATH_V12_8=<dir>` before building.

Other GPUs. The official wheel is built with `CUDA_ARCH_LIST=Common`, which CMake's FindCUDA turns into
sm_53 ... sm_86 plus compute_86 PTX (`cuobjdump --list-elf/--list-ptx`). A newer GPU (the store PC's
RTX 5060 Ti, sm_120) therefore runs the driver's JIT of that PTX, and the build that matches it is
`-Arch '8.6+PTX'`, not the GPU's own arch. `ptx_compare.py` checks a build against the official DLL
kernel by kernel: the store PC build (862d77c, built on another machine with `fetch_cuda.ps1` and VS 2022
Build Tools, moved over as a zip) has all 475 official kernels PTX-identical plus the fork's 28 new ones.
Results there (24.9, golden digest recorded from its own stock wheel): digest PASS; prod_equiv pipe8 (150
clips, `cpu_threads=1`) stock 49.4 s (16.6x) -> fork 25.2 s (32.2x), GPU busy 35.9 -> 23.3 s, rows_sha
ccbbc32e in every run (also stock batch8's). `cpu_threads=1` alone gains little on that CPU (48.8 -> 48.0 s).
GPU time there needs CUPTI 12.9 (12.6 and 12.8 answer CUPTI_ERROR_INVALID_DEVICE); `cupti.py` picks it by
compute capability. The full gate still has the RTX 2080's stock hashes and sm_75 probes built in.

Store PC, round 2 (25.9): 32.2x -> 35.4x (pipe8, 150 clips, 22.9 s), every hash equal to its stock wheel
(digest; pipe8 rows_sha ccbbc32e; exact2 6d5d58a2, 10.1 -> 15.5x; bench 60/90/118). There cuBLAS 12.9.2 runs
sm80 CUTLASS kernels with simple orders, recovered on the device's own cuBLAS (0 mismatches): decoder Dense
= one m16n8k16 chain over k, at k 5120 and 19..48 rows 3 serial split-K slices of 1728 (hmma_probe.cu);
attention scores = one chain, then half(alpha * acc) (qk_hmma_probe.cu); attention output = 16-key chain
with the residue of the 64-key tiles first (av_hmma_probe.cu). Changes:
- exact_attention.cuh: the encoder's self-attention in one kernel. Scores and probabilities stay in shared
  memory, keys and values are pre-arranged in fragment order (exact_attention_layout.cuh), the output is
  written heads-combined (combine_heads skips its transpose). exact_attention_check.cu: 0 mismatches at
  batch 20..160, 1.56x the three ops; 32.2 -> 34.3x.
- runner/features.py: the batched pipeline's log-mel on a thread pool (0.6 s of 150 clips with the GPU idle).
- The decoder state compacted in place when clips finish (the memory keys and values of every layer were
  copied whole into new buffers at each finish); Conv1D bias + GELU on 16-byte vectors. 35.0 -> 35.4x.
Tried, not kept: hmma_gemm.cuh (decoder Dense replica, exact on every shape, no faster than cuBLAS with the
weights coming from DRAM, slower in production); an encode-ahead queue of 2-3 batches (slower).

Store PC, round 3 (25-26.9): 35.4x -> 38.3x (pipe8, 150 clips, 21.2-21.3 s), digest PASS, rows_sha ccbbc32e.
**Deployed 26.9 as 55f83c7** (D:\ct2build\pyct2 + this runner in the production root; nothing started).
- On by default: residual add + next pre-norm in one kernel (cuda/residual_norm.cu); row softmax divisions as
  fma steps from one reciprocal; exact_attention stepping its shared-memory places, and fed straight from the
  fused projection (exact_attention_qkv); the decoder's cross-attention in one kernel (ops/cross_attention.cuh,
  only where cuBLAS's arithmetic was matched), with an L2 prefetch 4 steps ahead (CT2_CROSS_AHEAD); one cuBLAS
  handle per stream of a thread; runner: PIPE_ORDER=desc and the fallback ladder inline.
- Off by default (opt in): the encoder GEMM's CUTLASS replica (CT2_ENC_GEMM=cutlass, exact); the cross-attention
  query projection (CT2_CROSS_Q=1, exact, slower); SM partitions on green contexts (CT2_ENCODER_SMS,
  CT2_SM_PARTITION, slower); NVTX ranges (free without a profiler). Reverted: an L2 prefetch of decoder weights.
- CT2_CUDA_GRAPHS=1 (the decoder's steps as CUDA graphs, cuda/graph*.cc): 39.4x with identical rows, but a
  run faults ("unspecified launch failure", a Windows TDR) at a few specific batches in roughly 1 run in 3.
  compute-sanitizer (stream-ordered races, all 150 clips) is clean after the load fix (cecd31f); ruled out so
  far: graph updates, memcpy nodes, the pool's cross-stream reuse. Keep it off.
- Real units (scale/README.md): 30 store-PC units 28.9x vs stock production 12.2x, every deterministic row
  identical, fallback rows identical to stock under a fixed seed. Kernel probes on the device: softmax_check,
  ts_check, exact_attention_check all 0 (Smart App Control, which blocks new unsigned DLLs and exes, is off).
prod_kernels.py lists kernel time by name and launch shape and the GPU idle between kernels;
kernels/build_probe.ps1 -Gencode builds a probe for another target (compute_86 PTX for the store PC).
Measure: `python tools/turing/bench_whisper.py <sample_dir> <package parent dir>`.
Probes: `kernels/run_probe.ps1 <name> [args]` (nvcc sm_75, production's cuBLAS DLL): softmax_check,
qk_check, ts_check (gate), softmax_bench, qk_probe, qk_diff. Profiles: `host/nsys.ps1` (Nsight
Systems on the production engine) + `nsys_gaps.py`, `nsys_streams.py`, `nsys_steps.py` (decode steps,
`--grids`, `--pick=<q>`), `nsys_layer.py` (one encoder layer), `nsys_cpu.py`, `nsys_biggaps.py`.

Researched, not used: the exact summation orders of cuBLAS 12.9.2's fp16 kernels here, recovered by
comparing mma.sync m16n8k8 replicas with cuBLAS bit for bit (`kernels/gemm_probe.cu`, `attn_probe.cu`).
Decoder Dense at 5..40 rows: one chain over k in 8-groups (rows <= 15; N = 1280 at 20..30); two
chains alternating 32-k blocks, half(c0 + c1) (QKV and FFN1 at 20..40); split-K in 4 quarters each
rounded to half, then ((p0 + p1) + p2) + p3 in fp32 (N = 1280 at 35..40). Attention: QK one chain;
AV (k = 1500, encoder and decoder) residue-first groups (0-7, 8-15, 16-23, 24-27, then 28-35, ...).
`kernels/small_m_gemm.cuh` replicates the decoder GEMMs exactly (`gemm_check.cu`) but is no faster
than cuBLAS, and `lt_search.cu` found no faster cublasLt configuration with the same bits.

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
- Speed: `host/iterate.ps1` (build, digest, a discarded warm-up, then ONE base/next pair of prod_equiv
  pipe8 with `GPU_TIME=1` and a `verdict` line from `host/verdict.ps1`) or `host/equiv_ab.ps1`
  (environment settings). Why one pair: within one iterate run (same two builds, 24.9, 9 runs) the
  run-to-run noise was 1.74% on wall time and 0.18% on CUPTI GPU busy time, and the NIST sample size
  (e-Handbook 7.2.2.2) gives 64 runs per build to see a 1% change on wall time, 1 on GPU time. GPU
  time misses host-side and overlap gains (allocator, syncs, stream priorities): judge those with
  `-WallDelta <percent>`, which runs as many pairs as that change needs (16 for 2%). Before this the
  A/B was 79% of each ~11 min iteration (build 2, digest 1, A/B 8.6 min).
- Build: `build_windows.ps1` no longer forces the Python extension (64 s of every build): setup.py
  lists the public headers in `depends`, so it is rebuilt only when they or its sources change.

Host scripts (`host/`, for the Yarin layout; another host names its run folder in `<root>\work_dir.txt`;
each pulls the clone first and re-runs itself when that moved HEAD; start long ones detached, log in
`D:\ct2build\ab.log`):
`prod.ps1` shared helpers (pause/resume production, timed python runs); `gate.ps1` the full gate;
`digest.ps1` the fast check; `iterate.ps1` + `verdict.ps1` the dev loop; `equiv_ab.ps1` environment
A/Bs; `ab.ps1` builds and A/Bs with production
paused; `run_paused.ps1` runs one tool paused; `deploy.ps1` swaps `pyct2-next` into `pyct2` and ends
with Resume-Production, which STARTS the production supervisor.

Results on the RTX 2080, production engine (`prod_equiv.py` pipe8, 150 real clips), every hash equal
to stock (pipe8 262ababd, exact2 a83ba880; bench 6a0ac320 / 25b0f78a / 67bd5eeb):
- stock 11.5x realtime (13.1x with `cpu_threads=1`)
- softmax + cross-attention scores (724ae02) 14.0x, `cpu_threads=1` 15.5x, `add_range` 17.7x,
  timestamp rule 19.3x, DisableTokens on the GPU 19.3x, allocator (71841fb) 20.0x (40.6 s).
- Changes 7-12, measured in the daytime (the host's other apps kept about 3 of 6 cores busy, so the
  baseline took 47.5 s instead of 40.6), 71841fb vs e1a636e alternated: 47.5 -> 40.65 s wall, 37.9
  -> 32.8 s GPU busy; ratio 0.856, i.e. about 23.4x at the night-time 20.0x. Steps (wall): softmax
  45.2, split heads 44.6, deferred reorder 42.6, BiasAdd 42.5, timestamp rule 42.1, priorities 40.8.
