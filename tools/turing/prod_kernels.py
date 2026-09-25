"""Where the production engine's GPU time goes: prod_equiv.py's run (same engine, clips and mode, one warm-up
first) traced with kernel names; totals per kernel and per launch shape, and per stream busy time and overlap.
usage: prod_kernels.py <sample_dir> <engine_dir> <mode> [ctranslate2 package parent dir] [top N, default 25]"""
import os, sys, time, collections
from concurrent.futures import Future, ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
SAMPLE, ENGINE, MODE = sys.argv[1:4]
common.init(sys.argv[4] if len(sys.argv) > 4 else None)
TOP = int(sys.argv[5]) if len(sys.argv) > 5 else 25
sys.path.insert(0, ENGINE)
from faster_whisper import WhisperModel
from engine import transcribe_unit
from cupti import Tracer, short, busy

clips = common.sample_clips(SAMPLE)
model = WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cuda", compute_type="default",
                     num_workers=1 + int(MODE.startswith("pipe")) + 1, cpu_threads=1)
pool = ThreadPoolExecutor(max_workers=1)
run = lambda: [r.result() if isinstance(r, Future) else r for r in transcribe_unit(model, clips, MODE, pool)]
run()
tracer = Tracer(os.path.join(ENGINE, "cupti"))
tracer.start(); t = time.time(); run(); wall = time.time() - t; tracer.stop()
recs = tracer.records                              # (name, stream, start, end, grid, block, smem)
total = sum(r[3] - r[2] for r in recs)
print(f"wall {wall:.2f} s | kernels {len(recs)} | kernel time {total / 1e9:.2f} s | "
      f"GPU busy {busy([r[2:4] for r in recs]) / 1e9:.2f} s")
streams = collections.defaultdict(list)
for r in recs:
    streams[r[1]].append(r[2:4])
per = {s: busy(iv) for s, iv in streams.items()}
print("per stream busy: " + ", ".join(f"s{s} {v / 1e9:.2f} s ({len(streams[s])} kernels)"
                                      for s, v in sorted(per.items(), key=lambda x: -x[1])))


def table(title, key):
    t, c = collections.Counter(), collections.Counter()
    for r in recs:
        t[key(r)] += r[3] - r[2]; c[key(r)] += 1
    print(f"== {title}")
    for k, v in t.most_common(TOP):
        print(f"{v / 1e9:8.3f} s {100 * v / total:5.1f}% x{c[k]:7d} {v / c[k] / 1e3:8.1f} us  {k}")


table("per kernel", lambda r: short(r[0], 110))
table("per kernel and launch shape", lambda r: f"{short(r[0], 80)} g={r[4]} b={r[5]}")
# Idle GPU: every gap in the union of all streams' kernels, by the kernels on either side of it.
idle, count, end, last = collections.Counter(), collections.Counter(), None, None
for r in sorted(recs, key=lambda r: r[2]):
    if end is not None and r[2] > end:
        key = f"{short(last, 50)} -> {short(r[0], 50)}"
        idle[key] += r[2] - end; count[key] += 1
    if end is None or r[3] > end:
        end, last = r[3], r[0]
print(f"== idle GPU {sum(idle.values()) / 1e9:.2f} s in {sum(count.values())} gaps, by the kernels around them")
for k, v in idle.most_common(TOP):
    print(f"{v / 1e9:8.3f} s x{count[k]:7d} {v / count[k] / 1e3:8.1f} us  {k}")
pool.shutdown()
del model
import gc; gc.collect()
