"""GPU metrics of an nsys profile (`--gpu-metrics-devices=all`) by what was running: each metric sample is
put in the state of the GPU at its timestamp (encoder kernels only, decoder kernels only, both, or idle),
and every metric is averaged per state. A stream is the encoder's when its kernels average over 100 us
(the encoder's are milliseconds, the decoder's tens of microseconds).
usage: nsys_metrics.py <profile.sqlite>"""
import bisect, collections, sqlite3, sys

db = sqlite3.connect(sys.argv[1])
rows = db.execute("SELECT start, end, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
per_stream = collections.defaultdict(list)
for s, e, st in rows:
    per_stream[st].append((s, e))
enc_streams = {st for st, iv in per_stream.items() if sum(e - s for s, e in iv) / len(iv) > 100e3}


def union(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


enc = union(iv for st in enc_streams for iv in per_stream[st])
dec = union(iv for st in per_stream if st not in enc_streams for iv in per_stream[st])
enc_starts, dec_starts = [s for s, _ in enc], [s for s, _ in dec]


def active(starts, iv, t):
    i = bisect.bisect_right(starts, t) - 1
    return i >= 0 and iv[i][1] >= t


t0, t1 = min(r[0] for r in rows), max(r[1] for r in rows)
names = dict(db.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS"))
sums = collections.defaultdict(float)
counts = collections.Counter()
for t, mid, v in db.execute("SELECT timestamp, metricId, value FROM GPU_METRICS"):
    if not t0 <= t <= t1:
        continue
    state = {(True, True): "both", (True, False): "enc", (False, True): "dec",
             (False, False): "idle"}[(active(enc_starts, enc, t), active(dec_starts, dec, t))]
    sums[(state, mid)] += v
    counts[(state, mid)] += 1
states = ["enc", "dec", "both", "idle"]
some = next(iter(names))
total = sum(counts[(s, some)] for s in states) or 1
print(f"span {(t1 - t0) / 1e9:.2f} s | encoder streams {sorted(enc_streams)} | "
      + " ".join(f"{s} {100 * counts[(s, some)] / total:.1f}%" for s in states))
print(f"{'metric':44s}" + "".join(f"{s:>9s}" for s in states))
for mid, name in sorted(names.items()):
    vals = [sums[(s, mid)] / counts[(s, mid)] if counts[(s, mid)] else float("nan") for s in states]
    print(f"{name[:44]:44s}" + "".join(f"{v:9.1f}" for v in vals))
