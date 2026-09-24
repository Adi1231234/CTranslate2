"""Does decoding with the declared container (audio.audio_format) give the same samples as content
probing? For every clip of the given units: both decodes, compared bit for bit; clips that decode
only one way are listed. Run from the runner's root (hf_token.txt, units.json, HF cache).
usage: decode_check.py <unit_id> [unit_id ...]"""
import os, sys, json
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ["HF_HOME"] = os.path.join(ROOT, "hf")
import numpy as np, pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from units import DS, unit_id
from audio import audio_format, decode

fs = HfFileSystem(token=open(os.path.join(ROOT, "hf_token.txt")).read().strip())
units = {unit_id(u): u for u in json.load(open(os.path.join(ROOT, "units.json")))}
for uid in sys.argv[1:]:
    shard, rg = units[uid]
    t = pq.ParquetFile(fs.open(f"datasets/{DS}/{shard}", "rb")).read_row_group(rg, columns=["uuid", "audio"])
    same = differ = 0
    for uu, a in zip(t.column("uuid").to_pylist(), t.column("audio").to_pylist()):
        fmt = audio_format(a.get("path"))
        try:
            probed = decode(a["bytes"])
        except Exception as e:
            probed = None
        declared = decode(a["bytes"], fmt)
        if probed is None:
            print(f"{uid} {uu}: probing fails, declared {fmt} gives {len(declared)} samples")
        elif probed.dtype == declared.dtype and np.array_equal(probed, declared):
            same += 1
        else:
            differ += 1
            print(f"{uid} {uu}: DIFFERENT ({len(probed)} vs {len(declared)} samples)")
    print(f"{uid}: {same} identical, {differ} different (declared format {fmt})", flush=True)
