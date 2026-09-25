"""Production transcription of crowd-transcribe-v5 with the exact params. Resumable.

usage: python transcribe_run.py <front|back> <mode: exact2|batch8>
Producer thread streams row groups from HF and decodes audio; the GPU side never waits on I/O.
RUN_CACHE=<dir>: row groups kept in a local folder (fetch.py), so a benchmark repeats on the same bytes.
Each finished unit is written atomically to out/<unit_id>.jsonl, so a restart skips it.
"""
import os, sys, json, time, queue, threading
from concurrent.futures import Future, ThreadPoolExecutor
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import cudaenv  # noqa: F401
os.environ["HF_HOME"] = os.path.join(ROOT, "hf")
from huggingface_hub import HfApi, HfFileSystem
from faster_whisper import WhisperModel
from units import list_units, unit_id, my_order, should_stop, should_skip
from engine import transcribe_unit
from audio import audio_format, decode
from fetch import read_unit

DIRECTION, MODE = sys.argv[1], sys.argv[2]
OUT = os.environ.get("RUN_OUT") or os.path.join(ROOT, "out"); os.makedirs(OUT, exist_ok=True)
CACHE = os.environ.get("RUN_CACHE")             # fetched row groups kept on disk (fetch.py)
TOK_PATH = os.path.join(ROOT, "hf_token.txt")   # needed unless every unit is in the cache
TOK = open(TOK_PATH).read().strip() if os.path.exists(TOK_PATH) else None
fs, api = HfFileSystem(token=TOK), HfApi(token=TOK)

def log(msg):
    with open(os.path.join(ROOT, "progress.log"), "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

units_path = os.path.join(ROOT, "units.json")
if not os.path.exists(units_path):
    json.dump(list_units(fs, api), open(units_path, "w"))
units = my_order(json.load(open(units_path)), DIRECTION)

q = queue.Queue(maxsize=4)                      # ~4 row groups of decoded audio buffered
producer_failed = []
def producer():
    try:
        _produce()
    except Exception as e:                      # never leave the consumer blocked on q.get()
        producer_failed.append(e); log(f"PRODUCER CRASHED: {type(e).__name__}: {e}")
    q.put(None)

def _produce():
    handles = {}
    for u in units:
        uid = unit_id(u)
        why = should_stop(ROOT, uid)
        if why: log(f"STOP ({why}) before {uid}"); break
        if os.path.exists(os.path.join(OUT, uid + ".jsonl")) or should_skip(ROOT, uid): continue
        t = read_unit(fs, handles, u, uid, log, CACHE)
        if t is None:
            log(f"FETCH FAILED {uid}"); continue
        clips = []
        for uu, a in zip(t.column("uuid").to_pylist(), t.column("audio").to_pylist()):
            try: clips.append((uu, decode(a["bytes"], audio_format(a.get("path")))))
            except Exception as e: clips.append((uu, None)); log(f"decode fail {uu}: {e}")
        q.put((uid, clips))

threading.Thread(target=producer, daemon=True).start()
BATCHED = MODE != "exact2"
workers = 2 if MODE == "exact2" else 1 + int(MODE.startswith("pipe")) + 1   # +1 for async fallback
model = WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cuda", compute_type="default",
                     num_workers=workers, cpu_threads=1,   # the OpenMP threads of the default only spin
                     flash_attention=MODE.endswith("-fa"))
pool = ThreadPoolExecutor(max_workers=1) if BATCHED else None   # fallback clips off the critical path
done = queue.Queue()
stats = {"units": 0, "audio": 0.0, "t0": time.time()}

def writer():
    try:
        _write_loop()
    except Exception as e:                      # a failed fallback/write must not stall silently
        log(f"WRITER CRASHED: {type(e).__name__}: {e}"); os._exit(4)

def _write_loop():
    while True:
        item = done.get()
        if item is None: return
        uid, rows = item
        rows = [r.result() if isinstance(r, Future) else r for r in rows]
        tmp = os.path.join(OUT, uid + ".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, os.path.join(OUT, uid + ".jsonl"))
        stats["units"] += 1; stats["audio"] += sum(r["dur_s"] for r in rows)
        el = time.time() - stats["t0"]
        log(f"{uid} ok | units {stats['units']} | audio {stats['audio']/3600:.2f}h | {stats['audio']/el:.2f}x")

wt = threading.Thread(target=writer); wt.start()
log(f"START direction={DIRECTION} mode={MODE} units={len(units)} workers={workers}")
try:
    while True:
        item = q.get()
        if item is None: break
        uid, clips = item
        why = should_stop(ROOT, uid)
        if why: log(f"STOP ({why}) at {uid}, prefetched units dropped"); break
        done.put((uid, transcribe_unit(model, clips, MODE, pool)))
except Exception as e:                          # write what finished, then exit non-zero for a restart
    log(f"MAIN CRASHED: {type(e).__name__}: {e}"); done.put(None); wt.join(); os._exit(5)
done.put(None); wt.join()                       # drain: every processed unit is written before exit
log("FINISHED" if not producer_failed else "EXITING AFTER PRODUCER CRASH")
os._exit(3 if producer_failed else 0)  # non-zero -> supervisor restarts; resume skips done units
