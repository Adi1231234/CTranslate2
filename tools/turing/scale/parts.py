"""The real-data benchmark split in its two parts, each alone on the GPU: every cached unit through the batched
path with the fallback clips only recorded (their rows are not written anywhere), then the recorded clips through
the fallback ladder one by one. Reports each part's wall time, realtime factor and CUDA pool peak, and each
fallback clip's time and final temperature (random: sampling draws differ run to run).
usage: parts.py <cache dir> <units list> <runner dir> [mode, default pipe8] [ctranslate2 package parent dir]"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common
CACHE, LISTING, RUNNER = sys.argv[1:4]
MODE = sys.argv[4] if len(sys.argv) > 4 else "pipe8"
common.init(sys.argv[5] if len(sys.argv) > 5 else None)
sys.path.insert(0, RUNNER)
import pyarrow.parquet as pq
from faster_whisper import WhisperModel
from audio import audio_format, decode
from engine import transcribe_unit, _fallback


class Recorder:
    """Stands in for the fallback pool: keeps the clips, runs nothing."""
    def __init__(self):
        self.clips = []

    def submit(self, fn, model, uuid, wav):
        self.clips.append((uuid, wav))


units = [u for u in open(LISTING, encoding="utf-8").read().split() if u]
data = []
for uid in units:
    t = pq.read_table(os.path.join(CACHE, uid + ".parquet"))
    data.append([(uu, decode(a["bytes"], audio_format(a.get("path"))))
                 for uu, a in zip(t.column("uuid").to_pylist(), t.column("audio").to_pylist())])
audio_s = sum(len(w) for clips in data for _, w in clips) / 16000
model = WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cuda", compute_type="default",
                     num_workers=1 + int(MODE.startswith("pipe")), cpu_threads=1)
mem = common.mempool()
base = mem(common.USED_HIGH) >> 20                  # the weights (and whatever else is resident)
transcribe_unit(model, data[0], MODE, Recorder())   # warm-up
rec = Recorder()
mem(common.USED_HIGH, 0)
t0 = time.time()
for clips in data:
    transcribe_unit(model, clips, MODE, rec)
batched_s = time.time() - t0
batched_peak = mem(common.USED_HIGH) >> 20
mem(common.USED_HIGH, 0)
ladders = []
for uuid, wav in rec.clips:
    t = time.time()
    r = _fallback(model, uuid, wav)
    top = max((s["temperature"] for s in r["segments"]), default=None)
    ladders.append({"audio_s": round(len(wav) / 16000, 1), "seconds": round(time.time() - t, 1), "final_t": top})
fallback_s = sum(x["seconds"] for x in ladders)
print(json.dumps({"mode": MODE, "units": len(units), "audio_h": round(audio_s / 3600, 2), "weights_mb": base,
                  "batched": {"seconds": round(batched_s, 1), "realtime": round(audio_s / batched_s, 1),
                              "pool_peak_mb": batched_peak},
                  "fallback": {"clips": len(ladders), "seconds": round(fallback_s, 1),
                               "pool_peak_mb": mem(common.USED_HIGH) >> 20, "each": ladders},
                  "serial_realtime": round(audio_s / (batched_s + fallback_s), 1)}), flush=True)
del model
