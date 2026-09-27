"""Why batched decoding ends some clips early. Runs the production path (runner/engine.transcribe_unit, pipe8,
inline fallback, the production model settings) on units while split_recorder records each clip's decoded window,
then the sequential path on the clips whose batched window did not end in a single timestamp and on the clips of
an extra list, recording its windows too. Writes <out>/rows/<unit>.jsonl as the runner writes them (to compare
with a production run: compare.py) and <out>/windows.jsonl, one line per clip: its row's path and text, its
batched window, and where run, the sequential windows and text. Audio as in seeded.py (RUN_CACHE, else HF).
usage: cutoff_probe.py <runner_dir> <units list> <extra "unit uuid" list> <out dir> [ctranslate2 package parent]"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common
RUNNER, UNITS, EXTRA, OUT = sys.argv[1:5]
common.init(sys.argv[5] if len(sys.argv) > 5 else None)
sys.path.insert(0, RUNNER)
os.environ["HF_HOME"] = os.path.join(RUNNER, "hf")
import ctranslate2
from concurrent.futures import Future
from huggingface_hub import HfFileSystem
from faster_whisper import WhisperModel
import split_recorder as rec
from units import unit_id
from audio import audio_format, decode
from engine import transcribe_unit, _sequential, SR
from fetch import read_unit
from fallback import make_pool

ctranslate2.set_random_seed(1234)
wanted = [l.strip() for l in open(UNITS, encoding="utf-8") if l.strip()]
extra = {tuple(l.rstrip("\r\n").split(" ", 1)) for l in open(EXTRA, encoding="utf-8") if l.strip()}
token = os.path.join(RUNNER, "hf_token.txt")
fs = HfFileSystem(token=open(token).read().strip()) if os.path.exists(token) else None
units = {unit_id(u): u for u in json.load(open(os.path.join(RUNNER, "units.json")))}
cache, handles = os.environ.get("RUN_CACHE"), {}
os.makedirs(os.path.join(OUT, "rows"), exist_ok=True)
pool, _ = make_pool("inline", print)
model = WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cuda", compute_type="default",
                     num_workers=2, cpu_threads=1)       # as transcribe_run.py for pipe8 with inline fallback
strip = lambda c: {k: v for k, v in c.items() if k != "caller"}
counts = {"clips": 0, "not_single": 0, "seq_runs": 0}
with open(os.path.join(OUT, "windows.jsonl"), "w", encoding="utf-8") as wf:
    for uid in wanted:
        t = read_unit(fs, handles, units[uid], uid, print, cache)
        clips = [(u, decode(a["bytes"], audio_format(a.get("path"))))
                 for u, a in zip(t.column("uuid").to_pylist(), t.column("audio").to_pylist())]
        rec.calls.clear()
        rows = [r.result() if isinstance(r, Future) else r for r in transcribe_unit(model, clips, "pipe8", pool)]
        with open(os.path.join(OUT, "rows", uid + ".jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        batched = sorted((c for c in rec.calls if c["caller"] == "forward"), key=lambda c: c["offset"])
        good = [c for c in clips if len(c[1])]
        order = sorted(range(len(good)), key=lambda i: len(good[i][1]))   # engine._batch8's batching order
        assert len(batched) == len(good), (uid, len(batched), len(good))
        window = {}
        for k, c in zip(order, batched):
            assert abs(c["duration"] - len(good[k][1]) / SR) < 2 / SR, (uid, k)
            window[k] = strip(c)
        for k, ((uuid, wav), row) in enumerate(zip(good, rows)):
            line = {"unit": uid, "uuid": uuid, "dur_s": row["dur_s"], "path": row["path"], "text": row["text"],
                    "batch": window[k]}
            counts["clips"] += 1; counts["not_single"] += not window[k]["single_ending"]
            if (row["path"] == "batch8" and not window[k]["single_ending"]) or (uid, uuid) in extra:
                rec.calls.clear()
                s = _sequential(model, uuid, wav)
                line["seq"] = [strip(c) for c in rec.calls if c["caller"] == "generate_segments"]
                line["seq_text"] = s["text"]; counts["seq_runs"] += 1
            wf.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(uid, json.dumps(counts), flush=True)
print(json.dumps({"ctranslate2": ctranslate2.__file__, **counts}), flush=True)
