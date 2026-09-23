"""Where the GPU waits: the production engine (engine.transcribe_unit on the sample clips) under CUPTI
kernel + CUDA API tracing - the Nsight Systems timeline, reduced to numbers. Reports GPU busy/idle,
the idle gaps by length, what the host threads were inside during those gaps (a sync/copy call, a
launch, or no CUDA call at all = Python/CPU work), and the CUDA API totals per thread.
usage: timeline.py <sample_dir> <engine_dir> <mode> <out_log> [ctranslate2 package parent dir]"""
import os, sys, json, collections
from concurrent.futures import Future, ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
SAMPLE, ENGINE, MODE, OUT = sys.argv[1:5]
common.init(sys.argv[5] if len(sys.argv) > 5 else None)
sys.path.insert(0, ENGINE)
from cupti import Tracer, short
tr = Tracer(os.path.join(os.path.dirname(OUT), "cupti"), api=True)
import numpy as np
from faster_whisper import WhisperModel
from engine import transcribe_unit

meta, seen = [], set()
for m in json.load(open(os.path.join(SAMPLE, "meta.json"), encoding="utf-8")):
    if m["key"] not in seen:
        seen.add(m["key"]); meta.append(m)
clips = [(m["key"], np.load(os.path.join(SAMPLE, m["key"] + ".npy"))) for m in meta[:150]]
model = WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cuda", compute_type="default",
                     num_workers=1 + int(MODE.startswith("pipe")) + 1)
pool = ThreadPoolExecutor(max_workers=1)
run = lambda: [r.result() if isinstance(r, Future) else r for r in transcribe_unit(model, clips, MODE, pool)]
run()                                                           # warmup: allocator, cuBLAS, kernels
tr.start(); t0 = tr.now(); run(); t1 = tr.now(); tr.stop()


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


busy = merge((s, e) for _, _, s, e, *_ in tr.records if t0 <= s <= t1)
gaps = [(a[1], b[0]) for a, b in zip(busy, busy[1:])] + ([(t0, busy[0][0])] if busy else [])
wall, busy_ns = t1 - t0, sum(e - s for s, e in busy)
lines = [f"mode {MODE}: wall {wall / 1e6:.0f} ms, GPU busy {busy_ns / 1e6:.0f} ms ({100 * busy_ns / wall:.1f}%), "
         f"idle {(wall - busy_ns) / 1e6:.0f} ms, kernels {sum(1 for r in tr.records if t0 <= r[2] <= t1)}"]
buckets = [(0, 1e4, "<10us"), (1e4, 1e5, "10-100us"), (1e5, 1e6, "0.1-1ms"), (1e6, 1e7, "1-10ms"), (1e7, 1e12, ">10ms")]
for lo, hi, name in buckets:
    g = [e - s for s, e in gaps if lo <= e - s < hi]
    lines.append(f"  idle gaps {name:>8}: {len(g):7d} gaps, {sum(g) / 1e6:8.1f} ms")

api = [r for r in tr.api_records if r[3] >= t0 and r[2] <= t1]
inside, uncovered = collections.Counter(), 0
calls = sorted(api, key=lambda r: r[2])
j = 0
for s, e in gaps:                                               # host activity during each GPU gap
    while j < len(calls) and calls[j][3] < s - 50_000_000:      # skip calls that ended long before
        j += 1
    cover = []
    for name, tid, cs, ce in calls[j:]:
        if cs > e:
            break
        a, b = max(s, cs), min(e, ce)
        if b > a:
            inside[name] += b - a; cover.append((a, b))
    uncovered += (e - s) - sum(b - a for a, b in merge(cover))
lines.append(f"GPU-idle time by the CUDA call a host thread was inside (threads overlap, so sums can exceed "
             f"the idle total); no CUDA call on any thread: {uncovered / 1e6:.0f} ms")
for name, ns in inside.most_common(12):
    lines.append(f"  {ns / 1e6:9.1f} ms  {name}")
per = collections.defaultdict(lambda: collections.Counter())
cnt = collections.defaultdict(lambda: collections.Counter())
for name, tid, cs, ce in api:
    per[tid][name] += ce - cs; cnt[tid][name] += 1
for tid in sorted(per, key=lambda t: -sum(per[t].values())):
    lines.append(f"thread {tid}: CUDA API total {sum(per[tid].values()) / 1e6:.0f} ms, calls {sum(cnt[tid].values())}")
    for name, ns in per[tid].most_common(8):
        lines.append(f"  {ns / 1e6:9.1f} ms  x{cnt[tid][name]:7d}  {name}")
tot = collections.Counter(); kc = collections.Counter()
for n, _, s, e, *_ in tr.records:
    if t0 <= s <= t1:
        tot[short(n)] += e - s; kc[short(n)] += 1
lines.append("top kernels:")
for n, t in tot.most_common(10):
    lines.append(f"  {t / 1e6:9.1f} ms  x{kc[n]:7d}  {n}")
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
pool.shutdown()
del model                                       # release the model's worker threads while Python is alive
import gc; gc.collect()
