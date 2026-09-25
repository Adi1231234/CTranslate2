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
  3 workers before deploying.
