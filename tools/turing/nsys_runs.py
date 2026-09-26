"""The decoder's generate() calls in a pipelined nsys profile, one line each: a call's kernels run on one CT2
worker's stream, so a run of consecutive decoder kernels on one stream is a call. Per run: start, length,
kernels, the idle time before it (after the previous run) and its own idle gaps over 1 ms. Decoder streams:
kernels averaging <= 100 us. Runs under MIN kernels (default 1000) are counted, not listed.
usage: nsys_runs.py <profile.sqlite> [MIN]"""
import collections, sqlite3, sys

db = sqlite3.connect(sys.argv[1])
MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
S = dict(db.execute("SELECT id, value FROM StringIds"))
kern = sorted(db.execute("SELECT start, end, streamId, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"))
times = collections.defaultdict(list)
for s, e, st, _ in kern:
    times[st].append(e - s)
decoder = {st for st, d in times.items() if sum(d) / len(d) <= 100_000}
kern = [k for k in kern if k[2] in decoder]
print("decoder kernels per stream:", {st: len(times[st]) for st in sorted(decoder)})
runs = []
for s, e, st, n in kern:
    if runs and runs[-1][2] == st:
        runs[-1][1] = max(runs[-1][1], e)
        runs[-1][3].append((s, e))
    else:
        runs.append([s, e, st, [(s, e)], S.get(n, "?")])
print(f"{len(runs)} runs, {sum(1 for r in runs if len(r[3]) >= MIN)} with >= {MIN} kernels")
t0, prev, small = kern[0][0], None, 0
for s, e, st, ks, first in runs:
    if len(ks) < MIN:
        small += 1
        continue
    big = [(b[0] - a[1]) / 1e6 for a, b in zip(ks, ks[1:]) if b[0] - a[1] > 1e6]
    before = (s - prev) / 1e6 if prev is not None else 0.0
    print(f"{(s - t0) / 1e6:8.1f} ms stream {st:3d} {(e - s) / 1e6:8.1f} ms {len(ks):6d} kernels, idle before "
          f"{before:7.1f} ms ({small} short runs), gaps > 1 ms inside: {len(big)} = {sum(big):6.1f} ms "
          f"(max {max(big) if big else 0:5.1f}), first kernel {first[:24]}")
    prev, small = e, 0
