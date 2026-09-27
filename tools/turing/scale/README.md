# Scale check: a build against the stock wheel's production output

The release gate (../README.md) proves a build on 150 clips and a few targeted decodes. This check runs the
production runner itself on hundreds of real production units and compares every row, byte for byte, with the
rows the stock wheel wrote in the production run.

- `units_storepc.txt`: 325 store-PC units (32,030 clips) that the stock wheel transcribed on 23.9 after 01:42:20
  with the final engine (`batch8`, sorted batches, fallback on a side thread): all 180 units with a fallback row
  plus every third of the others. Earlier units ran other modes and cannot be compared row for row.
- `../host/scale_run.ps1 -Label <name> -Units <list> [-Mode pipe8] [-Pkg <build> | stock] [-Deadline ...]`:
  runs `runner/transcribe_run.py` (a fresh copy in `$R\verify\runner`, which needs `hf_token.txt`) on the list into `$R\verify\<name>`.
- `compare.py <ref_dir> <new_dir> [--sampled <list>]`: equal only if the JSON line is byte-identical. Rows
  decoded at a sampling temperature (fallback ladder past T=0, unseeded in production) differ between any
  two runs, so they are listed apart.
- `seeded.py <runner_dir> <list> <out.jsonl> [build]`: those clips through the sequential ladder on one worker
  with a fixed seed; the stock wheel twice (the control) and the build must give the same `rows_sha`.
- `speed.py <production progress.log> <scale progress.log>`: the two runs' wall time on the same units.
- `prefetch.py <runner_dir> <list> <cache dir> [threads]`: fetches a list's row groups into the run cache
  (`scale_run.ps1 -Cache`, `RUN_CACHE` for `seeded.py`), network only, so it can run beside a GPU job.
- `truncated.py <output dir>... > list.txt`: the `batch8` rows that stop over 1 s before their clip's end, as
  `seeded.py` lines. Batched decoding can end a clip early; the sequential path transcribes such clips to the end.
- `seeded.py` with `LADDER_LOG=<file>` also writes every attempt of the temperature ladder (`ladder_recorder.py`:
  temperature, token counts, compression ratio, log-prob, pass or fail, seconds, tokens); `ladder_report.py <file>
  capped|full [--brief]` sums it and marks where a repetition loop begins. The rows do not change with it.

## Result 25.9 (store PC, build G = e9d176b, runner with the feature cache, pipe8, cpu_threads=1)

- 90 of the 325 units ran before the store hours (units in list order, stopped between units): 8,859 clips,
  13.33 h. 8,820 rows byte-identical to production, **0 differences among deterministic rows** (including the
  3 fallback rows that passed at T=0). 52 rows decoded at a sampling temperature (39 of them differ from
  production, as any re-run's would).
- `seeded.py` on all 52: stock and build equal, `rows_sha` eeda189f78ccfbd9 (first 26) and 7a8fc4e7e71f7126
  (the other 26); their final temperatures cover the whole ladder (0.2 x12, 0.4 x5, 0.6 x2, 1.0 x32).
- `kernels/exact_attention_check`: TOTAL 0 over 207M outputs. `softmax_check` and `ts_check` do not start on the
  store PC: Windows App Control blocks those unsigned executables (their paths are covered by digest.py and
  this run; both passed on the 2080).
- **Speed on these units is not the sample's 35x:** production (stock, batch8, 2 workers) 10.7x, this run 9.9x.
  Half of the chosen units hold fallback rows (29% of production units), whose ladder runs clip by clip on one
  side worker; and the process used 7.4 GB of dedicated plus 1.0 GB of shared (system) GPU memory, i.e. WDDM
  paged it. Root cause not found yet: look at the pool release threshold (the build keeps freed memory) with
  3 workers before deploying. (Found: the async fallback, below.)

## Real-data benchmark 25-26.9 (store PC, `units_real.txt`: 30 cached units, 4.38 h, 10 fallback clips)

Wall time of `../host/scale_run.ps1 -Cache $R\verify\cache_real` (model load included) against the stock
production run on the same units (23.9, batch8, async fallback, 2 workers: 1291 s = 12.2x):

- build G, pipe8, async fallback: 747 s (21.1x), 760-860 MB paged to system memory; the pool release
  threshold at 0 changes nothing (764 s). The cause is the fallback's own worker: a ladder's memory beside a
  batch's overflows the 8 GB GPU. `parts.py` (each part alone): batched 475 s, the 10 ladders 134 s.
- build G, pipe8, `RUN_FALLBACK=inline`: 603 s (26.1x), 93 MB shared (none paged): inline is now the default.
- **build 55f83c7 (the deploy candidate), pipe8, inline, `PIPE_ORDER=desc`: 546 s = 28.9x, 2.37x stock**, 92 MB
  shared. `compare.py` IDENTICAL: 2,899 of 2,906 rows byte-identical, 0 deterministic differences; the 10 rows
  decoded at a sampling temperature: `seeded.py` stock twice and the build all `bb32a84f4f7b201d`.

## Both engines on the 101 units still missing from the crowd-v5 file (26-27.9, store PC)

Stock and the deployed build (55f83c7) through the production runner, pipe8, from one cache (`-Cache`, the same
bytes): stock 61 min (13.6x), build 27 min (30.3x). `compare.py`: 10,006 rows, 0 deterministic differences
(one of them a fallback row at T=0); 23 rows decoded at a sampling temperature, all equal under `seeded.py`
(stock twice and the build: `d58e9dfc63231d7c`). The stock output is the deliverable.

## Batch rows re-run on the sequential path (27.9, store PC)

`truncated.py` over the crowd-v5 file's batch rows and the 125 laptop units re-run with the build (pipe8, 31.8x):
4,607 rows (4,379 + 228), `seeded.py` with stock in 115 min (`970c93b630395ca7`). 3,805 came out identical and
802 changed; in 615 the batch text is a prefix of the sequential one (a clip cut short), and the sequential text is
closer to the human one in 724 rows vs 42 (mean CER 0.130 -> 0.098). The 1 s gap over-selects: 3,913 of the
sequential rows also end over 1 s early (silence at the clip's end), so it finds cut-offs, it does not prove one.

## Root cause of the cut-short batch rows (27.9, `cutoff_probe.py` on the 125 laptop units)

The probe re-ran the production path (rows identical to that morning's run: `compare.py` IDENTICAL) and recorded
every decoded window. 30 of 12,406 batched windows did not end in a single timestamp; the sequential path
transcribed 21 of the 24 `batch8` ones further (376 words), and 21 of the 23 cut rows found are among them.
- CTranslate2 generates at most 224 tokens per 30 s window (`whisper.cc`: `min(448 / 2, 448 - start_step)`, as
  OpenAI's `sample_len = n_text_ctx // 2`). Hebrew takes ~2.8 tokens a word, so a dense 29 s clip hits it
  mid-segment (25 of the 30; 19 lost text, 369 of the 376 words). The other 5 end with a timestamp pair, the
  paper's "segment continues past this window" signal (Radford et al. 2022, 2.3), near the clip's end.
- faster-whisper's `_split_segments_by_timestamps` then drops the unfinished segment and returns the frame to
  resume from. `generate_segments` (the sequential path, as OpenAI's `transcribe.py`) decodes a second window
  from there (22 of 24); `BatchedInferencePipeline.forward` ignores it (1.2.1 and master, 11.2025), so the rest of
  the clip is lost. The runner's fallback tests (empty, compression ratio, log-prob) do not see it.
- The 2 other cut rows ended in a single timestamp: the batched decode itself stopped earlier (fp16 batch drift).

## Yarin (RTX 2080, sm_75) after the store-PC work, and the full-context fix (27.9)

The store-pc library (611cf1c, built for 7.5) against Yarin's last tuned build e1a636e: digest and the whole release
gate PASS (probes 0, bench 60/90/118, pipe8 262ababd, exact2 a83ba880). 150 sample clips, 5 alternated rounds:
GPU time -2.5% and wall -2.2% in every round (765k vs 876k kernels); the fork runner vs Yarin's old one -1.1% wall;
PIPE_ORDER desc = asc there. The pool reserves ~0.7 GB more (7.3 vs 6.6 GB, used equal) but 30 real units show no
paging (shared peak 119 MB both): 18.41x vs 18.30x, `compare.py` IDENTICAL. The first run after idle on Yarin reads
the model from a cold HDD: two digests hit the 300 s limit (the next ones took 26 s), so warm it or discard it.

Full context (store-pc-fullctx 25cc1a32) on the same 30 units: 2,878 of 2,894 deterministic rows identical; 14 of the
16 others were cut short and now reach the clip's end (e.g. 66 -> 120 words). Cost 856 -> 934 s (+9%): the ladder
took 180 s vs 114 s, since a repetition loop now decodes up to 445 tokens per attempt, and 2 clips cut at 20-21 s
of 29 s now fail the thresholds at full length and go to the ladder (+23 s each). (Measured below: loops are the
smaller part.)

## Where the full context's ladder time goes, and why an early loop stop cannot keep every row (27.9, Yarin)

The 12 fallback clips of those 30 units through `seeded.py LADDER_LOG` (one worker, seed 1234; GPU time of the
generate() calls): full context 62 attempts, 176.3 s; store-pc 56 attempts, 128.5 s. Control: the full-context run
without the recorder gives the same `rows_sha` (d95b8813964a2c42).
- **Loops are 7 of the 62 attempts, 32.4 s.** Five are one 2.3 s clip (a token repeated from step 2); the others
  begin at steps 120 and 282. The other 55 attempts are normal text of 160-321 tokens that fail the compression
  ratio by a hair (2.40-2.69; log-prob fails 1 of 62). The 2 clips new to the ladder have no loop at all.
- **Why: a whole window of dense Hebrew compresses to ~2.4 by its length.** Over all 224,446 human crowd-v5
  transcripts the median ratio grows with length: 1.08 under 100 UTF-8 bytes, 2.06 at 400-499, 2.38 at 800-899;
  above 2.4 are 40% of the 800-899-byte ones and 64% of the 900-999-byte ones, most without a repeated word 3-gram.
  With the 224-token cap a window rarely holds that much; with the full context 7 of 14 windows fail every attempt
  (store-pc: 4 of 15).
- **An early stop changes rows.** (1) When every attempt fails, faster-whisper returns the best avg log-prob, among
  all attempts if none is under 2.4: seeded, 2 of the 12 rows are a 444-token loop (ratio 5.09 and 5.66), which
  any earlier stop changes. (2) A sampled attempt draws one Philox number per live row per step from per-row
  states that persist across calls (`src/cuda/random.cu`, `ops/multinomial_gpu.cu`), so stopping one changes every
  later sampled attempt. (3) A prefix does not bound the text still to come, so no stop short of the end proves
  that an attempt fails.
- **Upper bound of the gain:** stopping each loop two periods after it begins saves 24.5 s of the 176.3 s; with the
  batched path's +12 s at most that is under half of the +78 s.
