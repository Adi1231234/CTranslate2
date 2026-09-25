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
