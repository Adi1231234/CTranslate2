"""GPU time per CUDA stream from an Nsight Systems report exported to SQLite: each stream's busy time and
top kernels, and how long the streams actually ran kernels at the same time (the pipelined engine runs
the next batch's encoder on its own worker, so its own stream, next to the decoder).
usage: nsys_streams.py <report.sqlite> [top kernels per stream, default 15]"""
import sys, sqlite3, collections

db = sqlite3.connect(sys.argv[1])
top = int(sys.argv[2]) if len(sys.argv) > 2 else 15
S = dict(db.execute("SELECT id, value FROM StringIds"))
kern = db.execute("SELECT start, end, streamId, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()


def merged(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


t0, t1 = min(k[0] for k in kern), max(k[1] for k in kern)
by_stream = collections.defaultdict(list)
for s, e, st, n in kern:
    by_stream[st].append((s, e, n))
events = []                                        # +1/-1 per stream busy interval
for st, ks in by_stream.items():
    for s, e in merged((s, e) for s, e, _ in ks):
        events += [(s, 1), (e, -1)]
depth, last, conc = 0, t0, collections.Counter()
for t, d in sorted(events):
    conc[depth] += t - last; depth += d; last = t
print(f"window {(t1 - t0) / 1e6:.0f} ms; streams busy at once: "
      + ", ".join(f"{k}: {v / 1e6:.0f} ms" for k, v in sorted(conc.items())))
for st, ks in sorted(by_stream.items(), key=lambda x: -sum(e - s for s, e, _ in x[1])):
    tot, cnt = collections.Counter(), collections.Counter()
    for s, e, n in ks:
        tot[S.get(n, "?")] += e - s; cnt[S.get(n, "?")] += 1
    busy = sum(e - s for s, e in merged((s, e) for s, e, _ in ks))
    print(f"\nstream {st}: {len(ks)} kernels, busy {busy / 1e6:.0f} ms")
    for n, t in tot.most_common(top):
        print(f"  {t / 1e6:9.1f} ms  x{cnt[n]:7d}  {t / cnt[n] / 1e3:8.1f} us  {n[:90]}")
