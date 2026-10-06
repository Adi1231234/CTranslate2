"""Long recordings from a list (source<TAB>id<TAB>file in an audio folder), transcribed with the crowd-v5 parameters:
MODE=long through LongEngine (longform.py, the fork: many recordings at once), MODE=seq one after the other through
faster-whisper's own transcribe (the stock wheel's reference with RUN_STOCK_FULL_CONTEXT=1 and RUN_SEED, or the fork).
Rows to <out>/rows.jsonl in the list's order, each with its source and id. LONG_SECONDS=<n> (measurement only): each
recording's first n seconds. LONG_SHARD=<i>/<n>: every n-th recording from the i-th (several processes on one GPU).
Prints each recording's audio and time, then the rate from the model load to the end.
usage: python longform_run.py <list> <audio dir> <out dir>"""
import json, os, sys, time
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import cudaenv  # noqa: F401
from faster_whisper import WhisperModel, decode_audio
from faster_whisper.utils import download_model
import engine

listing, audio_dir, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
mode, cut = os.environ.get("MODE", "long"), int(os.environ.get("LONG_SECONDS", "0"))
items = [line.rstrip("\n").split("\t") for line in open(listing, encoding="utf-8") if line.strip()]
shard, shards = map(int, os.environ.get("LONG_SHARD", "0/1").split("/"))
items = items[shard::shards]
if os.environ.get("RUN_SEED"):                  # before the model: its workers seed their sampler states from it
    import ctranslate2
    ctranslate2.set_random_seed(int(os.environ["RUN_SEED"]))
model = WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cuda", compute_type="default",
                     num_workers=2 if mode == "long" else 1,   # long: the stream's loop holds one (longform.py)
                     cpu_threads=1)
if os.environ.get("RUN_STOCK_FULL_CONTEXT") == "1":
    from stock_context import full_context
    full_context(model)
t0, audio_s, rows = time.time(), 0.0, {}


def loader(name):
    def load():
        wav = decode_audio(os.path.join(audio_dir, name))
        return wav[:cut * 16000] if cut else wav
    return load


def done(i, row, started):
    global audio_s
    source, rid, _ = items[i]
    rows[i] = {"source": source, "id": rid, **row}
    audio_s += row["dur_s"]
    print(f"{source} {rid}: {row['dur_s']:.0f} s audio, {time.time() - started:.0f} s, "
          f"{len(row['segments'])} segments, total {audio_s / (time.time() - t0):.1f}x", flush=True)


if mode == "long":
    from longform import LongEngine
    long = LongEngine(model, download_model("ivrit-ai/whisper-large-v3-ct2", local_files_only=True))
    started = time.time()
    futures = [long.submit(f"{s}|{i}", loader(name)) for s, i, name in items]
    for k, future in enumerate(futures):
        done(k, future.result(), started)
    print(long.stats.line(), flush=True)
else:
    for k, (s, i, name) in enumerate(items):
        started, wav = time.time(), loader(name)()
        segments, _ = model.transcribe(wav, **engine.EXACT)
        done(k, engine._row(f"{s}|{i}", wav, list(segments), "seq"), started)
with open(os.path.join(out, "rows.jsonl"), "w", encoding="utf-8") as f:
    for k in range(len(items)):
        f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
wall = time.time() - t0
print(f"RESULT mode={mode} recordings={len(items)} audio_h={audio_s / 3600:.3f} wall_s={wall:.1f} "
      f"x_realtime={audio_s / wall:.2f}", flush=True)
