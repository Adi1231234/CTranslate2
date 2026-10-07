# The production runner on an AWS GPU box

How fast does the crowd-v5 pipeline (whisper large-v3, the exact decoding parameters, pipe8 + fallback, this fork)
run on an AWS GPU, and how much of the GPU does it leave idle? A throwaway EC2 box, driven from the laptop.

- `launch.ps1 [-Type g6e.xlarge] [-Az us-east-1a] [-Minutes 150]`: the box. Deep Learning Base AMI (Ubuntu 24.04),
  profile `docvoice-ec2-ssm`, tagged `Project=whisper-aws-bench`, and it terminates itself after `-Minutes`.
- `ssm.ps1 -Id <instance> -Script box\<step>.sh [-ScriptArgs ...]`: one step on the box through SSM, as root.
- `box/setup.sh <fork commit> <SSM parameter with the HF token> [units list]`: the venv (`requirements.txt`, the
  store PC's production versions), the fork, the Linux builds (`../linux/build.sh`, uploaded to
  `s3://docvoice-042984981008-code/whisper-aws-bench/`), the model, `units.json` checked against the production
  hash, and the units cached. The token is on the box only while the units are fetched.
- `box/run.sh <label> <stock | package> <runner dir> <units list> <processes> [VAR=value ...]`: one configuration,
  N runner processes at once (`MPS=1`: under NVIDIA MPS); one JSON line in `/opt/wb/results.jsonl`.

On Linux CTranslate2 loads cuBLAS with `dlopen("libcublas.so.12")`, so `run.sh` puts the venv's
`nvidia/cublas/lib` on `LD_LIBRARY_PATH` before the process starts (setting it from Python is too late).

`batch/` is the same benchmark as our own AWS Batch stack (`provision.py`, `image/build.py` in CodeBuild,
`submit.py <tag> experiments/<list> --fleet g6e|g6e2x [--privileged]`, `watch.py <job>`); results in
`s3://docvoice-042984981008-code/whisper-aws-bench/results/<job id>/`. Every round is a file in
`batch/experiments/` whose comment says what it tests and why. The laptop side:
- `race.py <region>/<job> ...`: the same round queued in several regions; the first copy to start runs, the others
  are cancelled (`follow_race.py <race.py's output>` then follows it with `watch.py`). Stacks (`WB_REGION`,
  `provision.py`): us-east-1, us-east-2, us-west-2, eu-north-1, eu-central-1, ap-south-1, ap-northeast-2.
- `fetch_results.py <job> <dir> [--out label ...]`: logs, rows, the GPU samples, and per configuration the wall and
  steady rates and the GPU energy with the rate at 330 W (a long run's rate, without the lone tail of a short one).
  `compare_ref.py <job> <dir> <label ...>`: the rows against the production reference (job d654b84b, `fullctx`).
  `gpu_samples.py`: power and clock histograms; `ladder_stats.py`: where the fallback ladders ended.
- `spend.py <day>`: what the day's jobs cost (on-demand price by region, plus a boot allowance).
- Kernel probes: `box/probes.sh` (`../linux/build_probes.sh`), `box/ncu_probe.sh` (one kernel's executed
  instructions by SASS opcode and line), `box/ncu_run.sh` (Nsight Compute on the runner); benches without the
  batched path: `box/ladder_bench.sh` (the fallback ladders alone, energy a ladder, rows digest),
  `box/encoder_bench.sh` (the encoder alone), `box/joined_check.sh` (sampled attempts alone against joined).
- Fleet `g7e` (RTX PRO 6000 Blackwell): compute environments created and then DISABLED, never run (Adi: measure
  only on g6e.xlarge, what production runs; dearer machines are not representative).

**Results 5.10.2026** (g6e L40S, us-east-1c, the 30 units of `../scale/units_real.txt` = 4.38 h, wall time with
model load, one run each; runs on two hosts of the same type differed by ~6%):
- Stock wheel 25.8x. The fork with the 224-token cap 42.3x, rows identical to stock (2,900 equal, 0 deterministic
  differences). Production (full context + `resume.py`) 37.9x, the lowest CER against the human text (0.0455 vs 0.0473).
- Several processes under MPS (`run.sh <n>`, `MPS=1`): 2 = 43.6x, 3 = 44.2x, 4 = 46.3x, rows identical to one
  process. Without MPS, 2 processes are slower than one (34.5x). On g6e.xlarge 4 processes use 96% of the 4 vCPUs;
  g6e.2xlarge (8 vCPU) with 4 is not faster (43.4x), and 6 or more run out of the 46 GB (about 8 GB a process).
  The GPU's memory controller is busy ~90% of the time from 2 processes on: about 45x is this pipeline's ceiling here.
- `CT2_CUDA_GRAPHS=1` is slower (36.4x) and changes 58 texts: a bug, not used. `pipe16`/`pipe32` (bigger batches)
  give 40.4x/42.2x but change ~22 deterministic texts and the CER gets worse (0.0459/0.0475).

**Later on 5.10 (branch ladder-probe; every run's rows IDENTICAL to production's, `../scale/compare.py`):** 46.3x ->
62.0x on the same 30 units (4 or 3 MPS processes; ~83x over the window where all processes run, ~70x estimated for a
long run). The batched path alone (`RUN_FALLBACK=skip`, measurement only) 86.6x, ~96x in that window.
- The store PC's exact kernels (fused encoder attention, fused cross-attention) hold on sm_89 too: the kernel probes
  (`../kernels/*`, run on the L40S through `batch/` script lines and `box/probes.sh`) found 0 mismatches; the
  cross-attention's tile residues differ by arch, so `ops/cross_attention_gpu.cu` has an sm_89 table. 53.7x.
- `CT2_CUDA_SCHEDULE=blocking` (host threads sleep while they wait for the GPU): 4 processes on 4 cores spent 2.2
  cores spinning in waits; CPU 96% -> 40%. 56-58x. (cub_caching allocator: 29x; 5 processes: no gain.)
- What limits it (`box/profile_metrics.sh`, privileged job, `../nsys_dram.py`): DRAM, ~72% read + 14% write at
  saturation, SM issue 12%. Decoder Dense products ~35-40% of the traffic, cross-attention 13-28% (at its byte
  floor), self-attention 5-20%, encoder ~15%, memory compaction copies ~5%.
- The 15 fallback clips (0.5%) cost ~28% of the GPU: ~40k decode steps at batch 1 (448 tokens x 6 temperatures)
  against ~25k batched steps; sampled hypotheses read 5 copies of the memory. Now one copy (pointer-array products,
  `../kernels/ptrbatch_probe.cu`: same bits; `../scale/sampled_check.py`, seeded: identical): +6%.
- `PIPE_GROUPS=k` / `group_size`: k batches in one beam search, each batch's products as that batch alone runs
  them (`src/cuda/clip_groups.h`); the products whose rows do not depend on the row count (`../kernels/rowinv2.cu`)
  in one call; the second feed-forward, whose cuBLAS split-K depends on the rows (`../kernels/ffn2_probe.cu`), in
  one pass with each group's split (`src/cuda/grouped_split_gemm.cuh`, `../kernels/grouped_split_check.cu`).
  L2 reuse of weights between separate calls did not happen under MPS.
- Memory slots (`src/cuda/memory_slots.h`): finished clips no longer compact the memory keys and values.

**The stream and seeded sampling (5.10 afternoon; three passes over the 30 units, `RUN_REPEAT=3` = 13.1 h, so the
model load and the last fallback ladders weigh little; rows IDENTICAL to production's):**
- `Whisper.open_stream` (runner `MODE=stream8`, `../runner/stream_engine.py`): batches decoded together, a batch
  joining as soon as there is room, across units; each batch exactly as `generate()` alone decodes it
  (`TransformerDecoder::decode_joint`, `BeamSearchRun`; `../scale/stream_check.py`: 800 of 800 clips identical,
  decoding 1.74x faster than batch by batch). A decoding step now has ~27 clips (9.2 before). The batched path
  2 x 6 batches: 106.6-108.5x against 90.2x for `PIPE_GROUPS=6`; 8 batches, 3 processes, a 32-query encoder
  attention (`CT2_EA_ROWS=32`, exact but slower alone): no gain.
- `sampling_seeds` (`src/cuda/row_random.h`; runner `RUN_FALLBACK_SEEDS=1`): every sampled hypothesis draws from a
  Philox stream of its own, seeded by its clip and its place in the ladder, so a clip samples the same alone or
  joined, and on every run (before, a row's random state was its place in the batch). With it, the full run with
  the fallback ladders joined (`RUN_FALLBACK_SAMPLING=batched`) gives the same rows, sampled ones too, as the ladders
  alone (2,906 of 2,906): exact, 80.8-82.7x, against 72.5-73.1x alone. A joined product must leave a group of one
  row to its own call (cuBLAS runs a gemv for one row; `rows_independent_product`).
- What limits it now (`box/profile_mps.sh` + gap analysis of the report): each process's decoding stream is idle
  ~31% of the time, in ~140k gaps under 0.1 ms between ~1,350 kernels and ~250 copies a step: per-op latency and
  host time, not DRAM (decoder weights 47 GB/s, cross-attention 199 GB/s). Fewer ops a step is the lever.
- The tiled exact GEMM (`src/cuda/tiled_split_gemm.cuh`, `../kernels/decoder_gemm_probe.cu`): every tile bit for
  bit cuBLAS, but cuBLAS is faster on every decoder product; only an opt-in (`CT2_DECODER_TILES`).
- `batch/` runs in other regions too (`WB_REGION`, the image and bucket stay in us-east-1) and `race.py` keeps the
  first of several copies of a job to start: g6e capacity was often sold out in us-east-1.

**What bounds it: the 350 W power limit (5.10 evening; rows IDENTICAL to production's in every change below):**
- The batched path runs at the L40S's power limit, 350 W, which is also the board's maximum (`box/power_limit.sh`);
  the SM clock sits at ~2100-2200 of 2520 MHz (8-14% of the samples at 2520: the moments it is not power-bound).
  The same run at 350 / 300 / 250 W: 110.6 / 100.2 / 80.0x
  (round32). So a kernel's cost is its energy, not its time: one process with 12 batches 100.4x against two with 6;
  side streams -3% (removed); the second feed-forward 45% faster alone, +0-1.3% on the run.
- Energy a decoding step (`../kernels/energy_probe.cu`, each kernel alone at production shapes, NVML's counter;
  9.2 of the ~10.8 J a step): cross-attention 2.54 J (DRAM-bound at 92% of peak, its 246 MB a clip a step are the
  floor), encoder attention 1.65 J (678M instructions a launch, two thirds the exact softmax's exp and division),
  encoder products 3.0 J (cuBLAS at 2.0-2.5 pJ a FLOP, power-limited at 141-178 TFLOPS), the second feed-forward 0.4,
  the decoder's other products 0.6, the memory keys and values 0.4, the GELU pass 0.25 J.
- Nsight Compute per kernel (`box/ncu_run.sh`): the cache reorder issued 346 instructions a 16-byte vector (64-bit
  index divisions) and is now a block per head row (`src/cuda/cache_reorder.cu`, `../kernels/cache_reorder_check.cu`);
  32-bit index math in the head split and the encoder attention's layout.
- Kept: the joint step's self-attention softmax in one launch (`src/cuda/softmax_parts.cu`; +1.5%, a row's arithmetic
  depends on its length only), each part's values product written in its rows of the context (no join kernel), the
  second feed-forward on 5 stages of 64 k (`src/cuda/grouped_split_gemm.cuh`). Opt-in, no gain: the encoder's first
  feed-forward on a CUTLASS replica with the bias and GELU in its epilogue (`CT2_ENC_GEMM=cutlass`; cuBLAS's s1688
  kernel has the 16-wide chain's bits on sm_89). Dropped: an encoder attention streaming keys and values once per
  128 queries in three passes (exact, 18% more instructions, -3%).
- `mma.sync m16n8k16` gives the same bits only when the k positions within a pair are swapped; any other order of
  the 16 changes them (`../kernels/mma_kperm_probe.cu`).
- Later (each A/B alternately on one machine, twice each; one machine's runs differ by up to ~2%): the joint step's
  cross-attention reading its queries from their Dense output, the bias added in the kernel (+1.4%); the encoder's
  first feed-forward on a pinned cuBLASLt algorithm with cuBLAS's bits and 20% less energy alone (`src/cuda/
  encoder_lt.cc`, `../kernels/encoder_algo_search.cu`; +0.35%, inside the noise); the cross-attention's L2 prefetch
  distance 1 on sm_89 (`../kernels/cross_ahead_probe.cu`: 74.2 against 79.5 mJ a launch; +0.3%). The batched path
  now 113-114x. No gain: the SM clock locked at 1800 / 1600 MHz (`box/clock_lock.sh`): 5% less energy an audio hour
  but -2.6% / -4.6% speed. Not possible here: ECC off (`box/ecc_mode.sh`): the GPU reset that applies it fails in a
  Batch container, the mode stays pending (ECC on: 46,068 MiB).
- The full exact run (`RUN_FALLBACK=batched`, seeded, round34 on pyct2-l41l): 74.2x with the sampled attempts alone,
  83.9x with them joined (rows IDENTICAL; 85.7x if the ladders still running at the end had run beside the batched
  path, as in a long run). The 15 fallback clips' ladders take ~24% of that run's energy, ~1 kJ a clip against
  16 J for a clip of the batched path: up to 6 attempts of up to 448 steps, mostly all 6 (7 of the 15 end at
  T = 1.0). Temperature variants (`sampling_temperatures`, runner `RUN_FALLBACK_SPECULATE=1`) sample a window's
  attempts in one search, each exactly what its own seeded call samples (`../scale/sampled_check.py`, `CHECK=variants`:
  75 of 75 attempts identical, alone 110.6 s against 44-48 s). The full run with them (round41, one machine):
  89.6x against 85.1x, every row identical between the two (8,718 of 8,718, the 39 sampled ones too); energy
  172.8 against 180.1 kJ, 90.3x at the batched path's power. The ladders are ~21% of all decoding row-steps
  (up to 448 steps x 25 sampled rows a window), so their energy is mostly the work itself.

**Where it stopped (5.10 night, Adi's call; every row IDENTICAL to production's; g6e.xlarge only):**
- Best package pyct2-l41p (6d2c3ee5, `s3://.../whisper-aws-bench/pyct2-l41p.tgz`) with runner-ee6fc199: the batched
  path alone 112.8-113.5x; the full run with the seeded, joined ladders and their temperature variants 89.6-90.5x
  (`RUN_FALLBACK_SPEC_CLIPS=2`: 1 clip a search 877-889 J a ladder, 2 788-832, 4 816, 8 867, 16 939; 4 also runs the
  GPU out of memory beside the batched path). At ~90x the 3,186 h take ~35 h on one g6e.xlarge, ~$66.
- Its production configuration (round44 `lv2`): `run.sh <label> pyct2-l41p runner-ee6fc199 <units list> 2 MPS=1
  MODE=stream8 STREAM_BATCHES=6 RUN_FALLBACK=batched RUN_FALLBACK_SEEDS=1 RUN_FALLBACK_SAMPLING=batched
  RUN_FALLBACK_SPECULATE=1 RUN_FALLBACK_SPEC_CLIPS=2 RUN_FALLBACK_GATHER=8 RUN_FALLBACK_GATHER_S=120
  CT2_CUDA_SCHEDULE=blocking`. Its deterministic rows equal the reference's; its sampled rows (temperature above 0,
  ~0.5% of rows) are seeded, so every run of it gives the same ones, but not the unseeded reference's.
- The whole benchmark cost ~$42 of g6e time (`spend.py 2026-10-05`: every whisper-bench job, all started that
  local day), plus CodeBuild and S3.
- The ceiling, measured: the encoder alone (`../scale/encoder_bench.py`) runs 263-274x at 347-349 W, ~7 J a clip;
  the batched path ~16 J a clip, so decoding is ~9 J (56%), most of it the cross-attention re-reading each clip's
  ~246 MB of memory keys and values every step. 140x needs ~13 J a clip, i.e. decoding a third cheaper with the same
  bits; no exact change found comes near (each below 1%).
- Measured and not adopted: 7 batches a stream (112.9 against 112.8x); the encoder's first feed-forward on the CUTLASS
  replica with bias and GELU fused at 256 x 128 (`CT2_ENC_GEMM=cutlass CT2_ENC_GEMM_CFG=256x128k32s3`, rows identical,
  +0-1%, inside the noise); the fork's CUDA graphs on the ladders (same bits now, more energy).
- On the branch, not packaged: a354d1bf, the encoder attention's loops split where its row layout jumps (bit checks
  TOTAL 0): 20% fewer instructions (Nsight Compute, `box/ncu_probe.sh`) but only 3% less energy for the kernel: its
  energy is the math and the shared-memory traffic, not the instruction count.

**Bit-for-bit against the stock wheel (6.10.2026, `batch/experiments/verify1.txt` and `verify2.txt`, jobs
994120ed in ap-northeast-2 and 1773cf47 in us-east-2, two L40S hosts; `python batch/verify_report.py 994120ed-8962-4581-
9632-fb10549b5603 1773cf47-ce82-49a6-a284-ee991b6f3e2a <dir>` prints every verdict; $3.70):** the original is
CTranslate2 4.8.2 from PyPI, unmodified, on the same GPU and the same cuBLAS (12.9.2.10; NVIDIA guarantees
run-to-run bits only for one toolkit on one architecture and SM count, and MMA bits differ across architectures,
so the reference has to be stock on the L40S itself). Stock decodes as many tokens as the fork through its own
max_length rule (`runner/stock_context.py`, `RUN_STOCK_FULL_CONTEXT=1`): no stock code changes, one argument.
- Stock twice (`RUN_SEED`, one worker, the inline fallback): 2,906 of 2,906 rows identical, the 13 sampled ones
  too, so equality below means something.
- pyct2-l41p in the same mode against stock, `compare.py --strict` (sampled rows counted): 2,906 of 2,906 on the
  30 units and 6,000 of 6,000 on 60 units no run had seen (`../scale/units_verify60.txt`, 8.7 h, a seeded random
  sample of the dataset): given the same random numbers, the fork's sampled rows are stock's too.
- The production configuration (runner-ee6fc199, stream, seeded joined ladders) against stock: every deterministic
  row identical (2,897 + 5,983); only sampled rows differ (its own per-clip seeds). Stock and production also equal
  the fullctx reference (d654b84b, another host) in every deterministic row.
- Every number the model outputs (`../scale/logits_check.py`): greedy decoding of 400 clips with every step's full
  vocabulary logits, 892,354,016 values, sha256 per clip: identical.
- The seeded draws (`../scale/seed_vs_stock.py`): stock's sampler on a fresh worker thread uses curand_init(seed,
  row, 0), the fork's seeded rows curand_init(seed, hypothesis, 0), so one-row calls get the same random numbers:
  75 calls (15 fallback clips x 5 temperatures), 16,557 sampled tokens and every score's bits identical.
- Kernel bit checks at 6d2c3ee5 (probes-6d2c3ee5): TOTAL 0 on every path production routes (encoder attention,
  softmax on every row length, softmax parts, cache reorder, pointer-array products, grouped split-K, encoder fc1
  replica and cuBLASLt pick, decoder tiles, timestamp rules; the sm_89 cross-attention residue table reproduced on
  a second host). The non-zero ones are the shapes the code excludes, i.e. the checks can see a difference: rowinv2
  at 1 row (gemv; `rows_independent_product` refuses groups of one row) and at 1280 x 5120 (310 of 320 row counts:
  the second feed-forward's split, hence `grouped_split_gemm`), selfattn_probe (363 pairs: self-attention runs per
  group), cross_sweep m = 1 (outside the fused route, m 2..8), hmma_check 1280 x 5120 (the store PC's replica,
  compiled into no sm_89 path: only `CT2_CROSS_Q`, off).
- What production changes against stock as shipped: stock with the 224-token cap against stock with the full
  context, same binary and seed: 15 of 2,906 rows, all 26-29 s clips whose text needs more than 224 tokens (14
  went to the fallback cut short, one lost its tail). That is the one intended difference (fork 25cc1a32).

**Long recordings (6.10.2026, the corpus transcribe list: 3,186 h, 93% in recordings of 34 min to 24 h;
`batch/experiments/long2.txt`, job f130110b, ap-northeast-2, $0.94; long1 hung, see `runner/longform.py`):** the
original long-form algorithm, faster-whisper's own sequential transcribe with the crowd-v5 parameters (30 s windows,
each conditioned on the text before it), on 48 YODAS v3 recordings and 2 Knesset plenums cut to 10 minutes (5.55 h).
- Stock, one recording after the other: 14.8x. The fork the same way: 18.9x, every row identical to stock with the
  same random numbers (8 recordings, 1,183 segments, sampled windows included).
- `runner/longform.py` (48 recordings at once: encoder calls joined, each window's beam search a batch of one in a
  stream, seeded ladders): 43.2x with the model load, 220-300 W (not power-bound), rows identical to stock up to the
  first sampled window (4 of 8 whole). Ladders are far more frequent than on the crowd clips: 7.7% of segments
  sampled, 24 of the 50 recordings.
- Features in blocks (`runner/chunked_features.py`): the FFT and the power do not depend on the block, the mel
  product does (OpenBLAS sums a column differently in a call of another width), so it stays the original's one call:
  byte-identical on the box's numpy 2.5.3 (12 of 12), and a 24 h plenum fits in ~17 GB instead of ~26.

**From 43x to 105x full run, 123x busy, on long recordings (6.10.2026, `batch/experiments/long3`..`long22`,
`final1`):** one measured cause at a time; every list50 row strictly identical to long6's dec48 rows
(`compare_long.py --strict`) after each step. Rates below are over seconds 40 to 400 of `long_stats.py`'s PROGRESS
lines (list120s, 48 threads). Careful: from f0 on, that span runs into the list's end (the recordings in progress
fall below 45 at ~300 s, 15 at 400 s), so it is not a steady rate. The steady rate is the span with every thread
in a recording ("fewest" >= 45 in the PROGRESS lines): f0 121.5x, m0 122.2x, n0 123.7x, o0 124.8x, final1 123.3x.
- Audio decoded in the process (a Python loop holding the GIL) starved the threads: `long_decode.py` worker
  processes, 43x to 51x. Ladders queued behind encodes on one dispatcher: they get lanes of their own
  (`LONG_LADDER_WORKERS`), now on the main model's workers (a second model instance only held memory).
- OOM at 48 threads (`CT2_CUDA_POOL_REPORT_S`): the cudaMallocAsync pool fragmented on self-attention caches that
  grew by Concat every step (a ladder's 25 rows copied ~2.5 GB a step) and on stale state caches. Fixed-capacity
  caches written in place (`CT2_CAPACITY_CACHES`, cuBLAS's bits with the capacity as batch stride:
  `capacity_stride_check` 0/22400), the state caches released when a window moves to the slots: 91.2x. The joint
  step's second feed-forward also fell back to a call per group above 16 groups (`gsg_run` now launches up to 64),
  and every window synced the host on its own (now three phases for all windows, one sync).
- Exact fused kernels, each a replica of cuBLAS's arithmetic recovered by probes: ladder cross-attention
  (`CT2_LADDER_CROSS`) 95.5x; TopK and LogSoftMax once for all windows (`joint_candidates`) 99.7x; the windows'
  self-attention, 2/3 of the GPU's time because the prompts make t 200-448 (`CT2_SLOT_ATTENTION`: a recipe per t,
  `selfattn_recipes.h` from `gen_selfattn_recipes.py`, the shared prompt read once; t < 32 stays on cuBLAS, whose
  recipe is ambiguous there; `slot_attention_check` 0/2502) 113.8x; the same for ladders on the capacity caches
  114.75x; their output sums spread over 8 lanes (`partial_sums.cuh`, they were 24% of the GPU's time) 116.1x; a
  ladder's single-row groups one joint call, then each row again alone (cuBLAS's gemv bits) 116.8x.
- The whole list to the end (`final1`, job 32c49777, us-east-2, $0.44; pyct2-l42o + runner-d83d636b): 120
  recordings cut to 10 minutes, 13.52 h of audio in 464.3 s = **104.8x** start to end (104.0x with the process
  start), the GPU 97% busy at 326 W on average. The first 40 s fill the 48 threads; from ~300 s fewer recordings
  are left than threads (each recording advances only ~3.5x while 48 share the GPU: a window ~8.5 s in the stream,
  each waiting for the one before), which costs ~160 s here. Rows: 120 of 120 strictly identical wherever an earlier run has them (50 to dec48, made before any of this
  work; 82 to q0, before any new kernel; 112 in all; 8 have no earlier row).
- What the corpus would run at: a work list of thousands of recordings has its start and end once, so close to
  the busy rate, 123x (3,186 h in ~26 h, ~$48 on one g6e.xlarge in us-east-1), not measured on recordings at their
  real length (10-minute cuts begin with short prompts more often, which is cheaper), and only if the longest
  recordings start first (a 24 h plenum at ~3.5x takes ~7 h; `longform.py` keeps the list's order today).
- At the recordings' real length (`full1`, job 1fd990e5, list120s uncut: 76.4 h, the list's order, stopped after 30
  minutes): ~115x while every thread holds a recording (seconds 160-880; the start takes 160 s, the long files'
  decoding), so the 10-minute cuts overstated the busy rate by ~6% (more of a cut recording's windows have a short
  prompt). The 58 recordings of 10 minutes or less: rows strictly identical to final1's. The host keeps >8 GB free
  once a recording's samples go after its features (runner 2cbab301); starting the longest first would hold ~49 GB
  of samples and features in this 28 GB job, so the corpus needs a cap on the hours in flight before it can.
- Where it stops: the GPU ~97% busy at ~312 W; ladders wait for a lane ~1,300-1,900 s per 400 s of run, and more
  lanes give nothing; 56 threads run out of memory. Without ladders (`prof1` k0, `LONG_LADDERS=skip`): 177x. Per CUDA
  stream (`prof1` p1, `box/nsys_streams.py`), in 15 s the ladder lanes ran 19.1 s of kernels against the joint
  stream's 11.3 s: cuBLAS's small-tile products 27% (a ladder step reads the decoder's weights on its own, ~1.47 GB),
  single-row gemv 17%, `lc_output` 13%, the second feed-forward 12%. `lc_output` reading a head's values once for a
  block of up to 32 rows (8b22a0f0, `ladder_cross_check` 0 of 7,810 against cuBLAS) looked free over seconds 40-280
  (`long23`: 124-125x against 126x on one host, 285 W instead of 324) but made a ladder call 12 s instead of 5.6 (40
  blocks for 142 SMs): the threads queued for their ladders (3,168 s of waits by 280 s against 671), a backlog that
  only a longer run would have shown in the rate. Reverted to one row a block (6d5d2607). A rate over a short span is
  not enough: read the STATS line's ladder waits too.
- Ladders in a stream (`LONG_LADDERS=stream`), every row strictly identical (long25, long26). In the windows' own
  stream (s1, s2) every ladder op waits its turn in the windows' steps (a ladder ~20 s instead of 5.6, 113x) and up to
  8 ladders' capacity caches (~1.8 GB each) ran the GPU out of memory. In a stream of their own beside the windows'
  (`LONG_LADDER_STREAM=1`, runner c4777ed7), up to 3 ladders together reading the weights once a step: **134.6x**
  against the lanes' 125.9x on the same host (`long26` t3, seconds 40-280; 2 together: 134.2x), a ladder ~27 s
  (~10 threads in ladders), 41.6 GB at the peak (long28, another host: 136.1-136.8x). Per CUDA stream there
  (`prof3`, the trace flushed every 200 ms: `--cuda-flush-interval`, prof2's lost the ladders' stream): the windows'
  stream 76% of 8 s (cross-attention 42%), the ladders' 69% (`lc_output` 24%, small-tile products 12%, slot
  attention 20%, single-row gemv 8%), the encoder 17%.
- Faster alone, slower in a run (every one exact): single rows in one launch (cbe37a07, `gemv_probe` recovered
  cuBLAS's one-row arithmetic: T 16, 32 or 8 partials t, t + T, ..., a tree from the halves; `single_rows_check`
  0 of 288, 2-16 rows in 0.2-0.6 of cuBLAS's time), `lc_output` 4 rows a block (152 us a launch instead of 455),
  the slot output's beams in one block (`slot_attention_check` 0 of 2,502): long28/long29/long32 129-132x and
  106.7x against 136x, a ladder 32-55 s against 26. Their fewer, longer blocks wait behind the other stream's
  kernels, and the run is bound by how long a recording waits for its ladder, not by the GPU's work. Kept, off by
  default (`CT2_SINGLE_ROWS=1`, `CT2_LC_ROWS`); the slot output is a beam a block again. Time a kernel in the run,
  never alone. The pieces of the ladders' stream:
  `GreedySearchRun` (`GreedySearch::search` runs it), `WhisperStream.submit_sampled`, the joint step's groups counted
  in rows with the greedy parts last, their attention run as their own search runs it (`layers/attention_sampled.cc`).

**Toward 170x (7.10.2026 night, Adi: 170x with every row identical, g6e.xlarge only; 10-minute cuts of list120s,
the rate over seconds 40-280, every comparison on one host, list50's rows strictly dec48's):**
- What bounds it: the GPU's capacity, not a latency. The windows' stream takes a window ~S(a + bN) for N windows in it
  (long34/35: a ~10 ms a step, b ~1.2 ms a window a step), so its throughput is nearly flat from 28 to 40 windows, and
  every change of who runs first only moves GPU time between the windows and the ladders: the windows' stream first
  (`LONG_WINDOW_PRIORITY=high`, w48 136.9x against z48's 136.8x), its cross-attention one launch a layer (l43b, c48
  136.2x), the windows in 2 streams (`LONG_WINDOW_STREAMS=2`, d48 135.6x). The GPU line now has the SM clock and the
  power cap's share: 92-99% of the samples power-capped, the clock 2,340-2,430 of 2,520 MHz. Without ladders
  (`LONG_LADDERS=skip`, rows change) 169.5x at 48 threads: the ladders take ~19% of the GPU, so 170x with them needs
  both the windows ~20% cheaper and the ladders much cheaper.
- Less work, every one exact (probes TOTAL 0, rows identical): the timestamp rules' device work of all of a step's
  searches at once (`src/joint_logits.h`, l43c: one disabled-tokens launch, one log-softmax, one reduction instead of
  ~150 launches a step), a beam's fork copies only the positions after the prompt (every slot gets the prompt at the
  expansion), `lc_output` a block per group of rows each on its own lanes and two dims a lane (l43e, l43g), single
  rows in one launch (`CT2_SINGLE_ROWS=1`, now a gain: +1.3-2.9%), the slot attention reading the prompt once for a
  part's beams (both mma chains take the beams as rows or columns of one mma, l43f: no gain). More threads: 56 and 60
  with the windows first (long38/39). Best measured: l43e with single rows, the windows first, 56 threads 147.9x
  (long40 r56, Seoul e9868788) against 141.3x for l42z the same way on another host; e60 147.3x against w56's 140.4x.
- More threads than windows (`LONG_STREAM_WINDOWS`, runner-378194a7): a thread waiting for its ladder holds only its
  encoder output (3.8 MB), so 72 threads with at most 40 windows decoding keep the stream full: s72 (l43g) 150.2x
  against r60's 148.6x (long41). The stream's throughput stays flat past ~35 windows (5.4 windows a second at 35 and
  at 40): the GPU's work is the bound. l43h (21174c9b: a lane's partials interleaved, each in its own order, probes
  TOTAL 0): h72 149.6x against s72's 147.8x on one host (long43, us-east-1 ca0ee5ea); 64 threads 147.2x.
- Not kept: the ladder's temperatures one at a time, the least work (`RUN_FALLBACK_SPEC_FIRST=1`: q72 145.6x against
  146.9x; `=5`: p72 131.5x at 293 W): the ladders' queue backs up, the stream empties and the GPU idles; 4-5 ladder
  batches, 80 threads with 44 windows and 64 threads uncapped run out of memory.
- Where the time goes now (`prof4`, l43g at 72 threads; `nsys` 2026.3.2 exports the report on a laptop): the windows'
  stream 81% busy, its cross-attention 41% of it at ~72% of the memory's bandwidth (it must read each window's 246 MB
  every step); the ladders' stream 51% (prof3: 69%), `lc_output` 0.69 s of 8 (1.31). Every row of every run that night
  strictly dec48's (17 runs).

**Beside other AWS work in the account** (the asr-training Batch queues): each touches only its own resources:
nothing named `asr-train*` here (their submit uses the newest `asr-train` job definition), nothing named
`whisper-bench*` there (their CancelJob is limited by IAM to jobs tagged Project=asr-training); everything tagged.
Capacity and the shared quota are first come, first served (Adi, 5.10): both may queue g6e.xlarge in any of the
regions above, never Japan. The account's on-demand G quota (5.10 night): 64 vCPU in us-east-1, eu-north-1 and
eu-central-1; 8 in us-east-2 and us-west-2 (requests for 32 open, July's for 64 closed with no increase there);
ap-northeast-2 8 and ap-south-1 0, requests for 64 open in both.
