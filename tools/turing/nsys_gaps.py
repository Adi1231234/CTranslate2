"""Where the GPU waits, from an Nsight Systems report exported to SQLite (nsys stats/export does it):
GPU busy vs idle, the idle gaps by length, which CUDA API call each host thread was inside during the
gaps (a synchronizing copy/sync, a launch, or none at all = Python/CPU work), per-thread API totals,
host<->device copies and the top kernels.
usage: nsys_gaps.py <report.sqlite>"""
import sys, sqlite3, collections

db = sqlite3.connect(sys.argv[1])
tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
S = dict(db.execute("SELECT id, value FROM StringIds"))
kern = db.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
api = sorted(db.execute("SELECT start, end, globalTid, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchall())


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


t0 = min(min(k[0] for k in kern), api[0][0])
t1 = max(max(k[1] for k in kern), max(a[1] for a in api))
busy = merge((s, e) for s, e, _ in kern)
gaps = [(t0, busy[0][0])] + [(a[1], b[0]) for a, b in zip(busy, busy[1:])] + [(busy[-1][1], t1)]
wall, busy_ns = t1 - t0, sum(e - s for s, e in busy)
out = [f"wall {wall / 1e6:.0f} ms, GPU busy {busy_ns / 1e6:.0f} ms ({100 * busy_ns / wall:.1f}%), "
       f"idle {(wall - busy_ns) / 1e6:.0f} ms, kernels {len(kern)}, CUDA API calls {len(api)}"]
for lo, hi, name in [(0, 1e4, "<10us"), (1e4, 1e5, "10-100us"), (1e5, 1e6, "0.1-1ms"), (1e6, 1e7, "1-10ms"),
                     (1e7, 1e13, ">10ms")]:
    g = [e - s for s, e in gaps if lo <= e - s < hi]
    out.append(f"  idle gaps {name:>8}: {len(g):7d} gaps, {sum(g) / 1e6:8.1f} ms")

inside, uncovered, active, k = collections.Counter(), 0, [], 0
for s, e in gaps:                                  # sweep: calls overlapping each idle gap
    while k < len(api) and api[k][0] <= e:
        active.append(api[k]); k += 1
    active = [c for c in active if c[1] >= s]
    cover = []
    for cs, ce, tid, nid in active:
        a, b = max(s, cs), min(e, ce)
        if b > a:
            inside[S[nid]] += b - a; cover.append((a, b))
    uncovered += (e - s) - sum(b - a for a, b in merge(cover))
out.append(f"GPU idle while a host thread was inside a CUDA call (threads overlap, sums can exceed the idle "
           f"total); idle with NO CUDA call on any thread (host CPU/Python work): {uncovered / 1e6:.0f} ms")
for name, ns in inside.most_common(12):
    out.append(f"  {ns / 1e6:9.1f} ms  {name}")

per, cnt = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
for s, e, tid, nid in api:
    per[tid & 0xFFFFFF][S[nid]] += e - s; cnt[tid & 0xFFFFFF][S[nid]] += 1
for tid in sorted(per, key=lambda t: -sum(per[t].values())):
    out.append(f"thread {tid}: CUDA API time {sum(per[tid].values()) / 1e6:.0f} ms in {sum(cnt[tid].values())} calls")
    for name, ns in per[tid].most_common(8):
        out.append(f"  {ns / 1e6:9.1f} ms  x{cnt[tid][name]:7d}  {name}")
if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
    for kind, n, b, t in db.execute("SELECT copyKind, count(*), sum(bytes), sum(end - start) "
                                    "FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"):
        out.append(f"memcpy kind {kind}: {n} copies, {b / 1e6:.1f} MB, {t / 1e6:.1f} ms")
tot, kc = collections.Counter(), collections.Counter()
for s, e, nid in kern:
    tot[S[nid]] += e - s; kc[S[nid]] += 1
out.append("top kernels:")
for n, t in tot.most_common(12):
    out.append(f"  {t / 1e6:9.1f} ms  x{kc[n]:7d}  {n[:100]}")
print("\n".join(out))
