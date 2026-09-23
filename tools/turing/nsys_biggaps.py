"""The long GPU idle gaps one by one, from a sampled Nsight Systems report exported to SQLite: when each
gap starts, the kernels around it, and per host thread the CUDA call it was inside and its innermost
python frames during the gap (CPU samples, --python-sampling=true).
usage: nsys_biggaps.py <report.sqlite> [min_gap_ms=5]"""
import sys, sqlite3, bisect, collections

db = sqlite3.connect(sys.argv[1])
MIN_GAP = float(sys.argv[2] if len(sys.argv) > 2 else 5) * 1e6
S = dict(db.execute("SELECT id, value FROM StringIds"))
pid = db.execute("SELECT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 1").fetchone()[0]
kern = sorted((s, e, S.get(n, "?"), st) for s, e, n, st in
              db.execute("SELECT start, end, shortName, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL"))
busy, last = [], []                               # merged busy intervals, index of their last kernel
for i, (s, e, _, _) in enumerate(kern):
    if busy and s <= busy[-1][1]:
        if e > busy[-1][1]:
            busy[-1][1], last[-1] = e, i
    else:
        busy.append([s, e]); last.append(i)
t0 = busy[0][0]
gaps = [(busy[j][1], busy[j + 1][0], last[j]) for j in range(len(busy) - 1) if busy[j + 1][0] - busy[j][1] >= MIN_GAP]
calls = collections.defaultdict(list)
for s, e, tid, nid in db.execute("SELECT start, end, globalTid, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
    calls[tid].append((s, e, S.get(nid, "?")))
for v in calls.values():
    v.sort()
by_time = sorted((t, i, g) for i, t, g in db.execute(
    "SELECT id, start, globalTid FROM COMPOSITE_EVENTS WHERE globalTid / 16777216 = ? / 16777216", (pid,)))
names = {t: S.get(n, "") for n, _, t in db.execute("SELECT nameId, priority, globalTid FROM ThreadNames")}


def py_frames(sid):
    rows = db.execute("SELECT module, symbol FROM SAMPLING_CALLCHAINS WHERE id = ? ORDER BY stackDepth", (sid,))
    out = [S.get(sym, "?") for mod, sym in rows if S.get(mod, "").endswith(".py") or "python" in S.get(mod, "").lower()]
    return " <- ".join(f[:40] for f in out[:3]) or "(no python frame)"


def api_at(tid, t):
    v = calls.get(tid, [])
    i = bisect.bisect_right(v, (t, float("inf"), "")) - 1
    return v[i][2] if i >= 0 and v[i][1] >= t else "host code"


print(f"{len(gaps)} gaps >= {MIN_GAP / 1e6:.0f} ms, {sum(b - a for a, b, _ in gaps) / 1e6:.0f} ms, "
      f"window {(busy[-1][1] - t0) / 1e6:.0f} ms")
for a, b, k in gaps:
    nxt = bisect.bisect_left(kern, (b,))
    print(f"\n@{(a - t0) / 1e6:9.1f} ms  gap {(b - a) / 1e6:6.1f} ms  after {kern[k][2][:40]} (stream {kern[k][3]})"
          f"  before {kern[nxt][2][:40]} (stream {kern[nxt][3]})")
    lo, hi = bisect.bisect_left(by_time, (a,)), bisect.bisect_right(by_time, (b,))
    per = collections.defaultdict(collections.Counter)
    for t, sid, tid in by_time[lo:hi]:
        per[tid][f"{api_at(tid, t)} | {py_frames(sid)}"] += 1
    for tid, c in sorted(per.items(), key=lambda x: -sum(x[1].values()))[:5]:
        top = "; ".join(f"{n}x {w}" for w, n in c.most_common(2))
        print(f"   thread {tid & 0xFFFFFF:6d} {names.get(tid, '')[:12]!r:14} {sum(c.values()):4d} samples: {top}")
