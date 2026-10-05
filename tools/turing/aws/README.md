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
`submit.py <tag> experiments/<list> --fleet g6e|g6e2x`, `watch.py <job>`); results in
`s3://docvoice-042984981008-code/whisper-aws-bench/results/<job id>/`.

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

**Beside other AWS work in the account** (the asr-training Batch queues): a standalone box, never their queues;
no resource named `asr-train*` (their submit uses the newest `asr-train` job definition); another AZ than their
running box; everything tagged.
