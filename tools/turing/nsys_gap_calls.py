"""What the host threads call during the decoder's long gaps in a pipelined nsys profile (--trace=cuda is
enough, no sampling): for each decoder-stream gap of at least MIN ms, per thread the CUDA runtime calls made
inside it (count and time by name) and the longest stretch without any call (host code or blocked).
Decoder streams: kernels averaging <= 100 us.
usage: nsys_gap_calls.py <profile.sqlite> [min_gap_ms=1] [gaps to list=12]"""
import bisect, collections, sqlite3, sys

db = sqlite3.connect(sys.argv[1])
MIN = float(sys.argv[2] if len(sys.argv) > 2 else 1) * 1e6
LIST = int(sys.argv[3]) if len(sys.argv) > 3 else 12
S = dict(db.execute("SELECT id, value FROM StringIds"))
kern = sorted(db.execute("SELECT start, end, streamId, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"))
times = collections.defaultdict(list)
for s, e, st, _ in kern:
    times[st].append(e - s)
decoder = {st for st, d in times.items() if sum(d) / len(d) <= 100_000}
dec = [k for k in kern if k[2] in decoder]
gaps, end, last = [], None, None
for s, e, st, n in dec:
    if end is not None and s - end >= MIN:
        gaps.append((end, s, S.get(last, "?"), S.get(n, "?")))
    if end is None or e > end:
        end, last = e, n
calls = collections.defaultdict(list)
for s, e, tid, nid in db.execute("SELECT start, end, globalTid, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
    calls[tid].append((s, e, S.get(nid, "?").split("_v")[0]))
for v in calls.values():
    v.sort()
total = collections.Counter()
t0 = dec[0][0]
print(f"{len(gaps)} decoder gaps >= {MIN / 1e6:g} ms, {sum(b - a for a, b, *_ in gaps) / 1e6:.0f} ms")
for n, (a, b, before, after) in enumerate(gaps):
    show = n < LIST
    if show:
        print(f"\n@{(a - t0) / 1e6:9.1f} ms gap {(b - a) / 1e6:6.2f} ms after {before[:30]} before {after[:30]}")
    for tid, v in calls.items():
        lo, hi = bisect.bisect_left(v, (a,)), bisect.bisect_left(v, (b,))
        inside = v[lo:hi]
        if not inside:
            continue
        by = collections.Counter()
        for s, e, name in inside:
            by[name] += min(e, b) - s
            total[name] += min(e, b) - s
        edges = [a] + [x for s, e, _ in inside for x in (s, e)] + [b]
        free = max(edges[i + 1] - edges[i] for i in range(0, len(edges) - 1, 2))
        if show:
            top = ", ".join(f"{k} {t / 1e6:.2f}ms x{sum(1 for *_, m in inside if m == k)}" for k, t in by.most_common(3))
            print(f"   thread {tid & 0xFFFFFF:6d}: {len(inside):5d} calls, longest free {free / 1e6:6.2f} ms; {top}")
print("\ntime inside CUDA calls during all these gaps, by call:")
for k, t in total.most_common(10):
    print(f"  {t / 1e6:9.1f} ms  {k}")
