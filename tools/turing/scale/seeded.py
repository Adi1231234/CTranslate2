"""Seeded re-run of the clips compare.py lists as decoded at a sampling temperature, where no two production
runs agree: each clip goes through the runner's sequential path (the full temperature ladder, as the fallback
runs it) on one CTranslate2 worker with a fixed random seed, so a build gives the same rows on every run and two
builds can be compared byte for byte. Audio comes from HF as in the runner (its hf_token.txt and units.json).
usage: seeded.py <runner_dir> <list file> <out.jsonl> [ctranslate2 package parent dir]    N_CLIPS=<n>: the first n"""
import os, sys, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common
RUNNER, LISTING, OUT = sys.argv[1:4]
common.init(sys.argv[4] if len(sys.argv) > 4 else None)
sys.path.insert(0, RUNNER)
os.environ["HF_HOME"] = os.path.join(RUNNER, "hf")
import ctranslate2
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from faster_whisper import WhisperModel
from units import DS, unit_id
from audio import audio_format, decode
from engine import _sequential

ctranslate2.set_random_seed(1234)               # every worker thread's sampler starts from this seed
wanted = {}
lines = [line.rstrip("\r\n") for line in open(LISTING, encoding="utf-8") if line.strip()]
for line in lines[:int(os.environ.get("N_CLIPS", len(lines)))]:   # N_CLIPS: the first n of the list
    uid, uuid = line.split(" ", 1)                                  # crowd-v5 uuids hold spaces
    wanted.setdefault(uid, []).append(uuid)
fs = HfFileSystem(token=open(os.path.join(RUNNER, "hf_token.txt")).read().strip())
units = {unit_id(u): u for u in json.load(open(os.path.join(RUNNER, "units.json")))}
model = WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cuda", compute_type="default",
                     num_workers=1, cpu_threads=1)          # one worker: one sampler, one order of draws
with open(OUT, "w", encoding="utf-8") as f:
    for uid in sorted(wanted):
        shard, rg = units[uid]
        t = pq.ParquetFile(fs.open(f"datasets/{DS}/{shard}", "rb")).read_row_group(rg, columns=["uuid", "audio"])
        audio = dict(zip(t.column("uuid").to_pylist(), t.column("audio").to_pylist()))
        for uuid in wanted[uid]:                    # a uuid repeated in a unit: every copy decodes the same
            a = audio[uuid]
            row = _sequential(model, uuid, decode(a["bytes"], audio_format(a.get("path"))))
            f.write(json.dumps({"unit": uid, **row}, ensure_ascii=False) + "\n")
digest = hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:16]
print(json.dumps({"ctranslate2": ctranslate2.__file__, "clips": sum(map(len, wanted.values())), "rows_sha": digest}),
      flush=True)
del model
