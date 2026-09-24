# crowd-v5 production runner

The engine that transcribed ivrit-ai/crowd-transcribe-v5 (224,538 clips) on Yarin's RTX 2080 and the
store PC, kept here as it ran (commit 01a8804) and changed only through this repo. It lives in
`D:\wsbench-tmp` on Yarin next to its venv, `hf` cache, `units.json` (the 2268 units = shard row
groups) and `out\<unit>.jsonl`; copy these files there after a pull.

- `supervise.ps1 -root D:\wsbench-tmp -dir back -mode pipe8 [-pythonpath <ctranslate2 build>]`
  restarts `transcribe_run.py` on a crash or a 20-minute stall. Without `-pythonpath` it runs the
  venv's stock wheel.
- `transcribe_run.py`: a producer thread streams row groups from HF (needs `hf_token.txt`) and
  decodes the audio; `engine.transcribe_unit` runs the exact decoding parameters (`pipe8`: sorted
  batches of 8 with the next batch's encoder overlapped, clips failing the thresholds re-run with
  the full temperature ladder on a side thread). Units whose output exists are skipped.
- `stop.json` (re-read before every unit): `skip_units`, `only_units`, `stop_before_unit`, `deadline`.
- `audio.py`: the decode (container from the declared file extension); `decode_check.py <units>`
  compares it with content probing, bit for bit. `RUN_OUT=<dir>` writes the outputs elsewhere.
- `redo_units.txt`: the 382 units Yarin wrote on 23.9 between 14:25 and 18:38 with the faulty
  softmax build (see ../README.md, incident); rerun with `only_units` = this list.
