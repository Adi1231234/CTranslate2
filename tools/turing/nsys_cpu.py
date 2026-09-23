"""What the host CPU does while the GPU is idle, from an Nsight Systems report with CPU sampling
(--sample=process-tree, --python-sampling=true), exported to SQLite: every CPU sample of the profiled
process taken during a GPU idle gap of at least MIN_GAP, attributed to its leaf function and to the
innermost ctranslate2 / python frame of its call stack.
usage: nsys_cpu.py <report.sqlite> [--schema] [--stacks N]"""
import sys, sqlite3, bisect, collections

MIN_GAP = 100_000                                  # ns
db = sqlite3.connect(sys.argv[1])
if "--schema" in sys.argv:
    for (t,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        n = db.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        cols = [c[1] for c in db.execute(f"PRAGMA table_info({t})")]
        print(f"{t} ({n} rows): {', '.join(cols)}")
    sys.exit()
S = dict(db.execute("SELECT id, value FROM StringIds"))
pid = db.execute("SELECT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 1").fetchone()[0]
kern = sorted(db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"))
busy = []
for s, e in kern:
    if busy and s <= busy[-1][1]:
        busy[-1][1] = max(busy[-1][1], e)
    else:
        busy.append([s, e])
gaps = [(a[1], b[0]) for a, b in zip(busy, busy[1:]) if b[0] - a[1] >= MIN_GAP]
starts = [g[0] for g in gaps]
t0, t1 = busy[0][0], busy[-1][1]
states = dict(db.execute("SELECT id, name FROM ENUM_SAMPLING_THREAD_STATE"))
thread_names = {t: S.get(n, "") for n, _, t in db.execute("SELECT nameId, priority, globalTid FROM ThreadNames")}

inside, total = [], collections.Counter()
for sid, start, tid, st in db.execute("SELECT id, start, globalTid, threadState FROM COMPOSITE_EVENTS "
                                      "WHERE start BETWEEN ? AND ? AND globalTid / 16777216 = ? / 16777216",
                                      (t0, t1, pid)):
    total[tid] += 1
    i = bisect.bisect_right(starts, start) - 1
    if i >= 0 and start < gaps[i][1]:
        inside.append((sid, tid, st))
idle = sum(e - s for s, e in gaps)
print(f"GPU idle in gaps >= {MIN_GAP // 1000} us: {idle / 1e6:.0f} ms of {(t1 - t0) / 1e6:.0f} ms; process CPU "
      f"samples {sum(total.values())}, {len(inside)} of them inside those gaps")
by_thread = collections.Counter(tid for _, tid, _ in inside)
for tid, n in by_thread.most_common(6):
    print(f"  thread {tid & 0xFFFFFF} {thread_names.get(tid, '')!r}: {n} samples in gaps, {total[tid]} overall")
print("  thread states in gaps:", dict(collections.Counter(states.get(st, st) for *_, st in inside)))

db.execute("CREATE TEMP TABLE sel (id INTEGER PRIMARY KEY)")
db.executemany("INSERT INTO sel VALUES (?)", [(sid,) for sid, _, _ in inside])
frames = collections.defaultdict(list)
for sid, sym, mod, depth in db.execute("SELECT c.id, c.symbol, c.module, c.stackDepth FROM SAMPLING_CALLCHAINS c "
                                       "JOIN sel USING (id) ORDER BY c.id, c.stackDepth"):
    frames[sid].append((S.get(mod, "?").split("\\")[-1], S.get(sym, "?")))
calls = collections.defaultdict(list)             # the CUDA API call each sampled thread was inside
for s, e, tid, nid in db.execute("SELECT start, end, globalTid, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME "
                                 "WHERE end >= ? AND start <= ?", (t0, t1)):
    calls[tid].append((s, e, S.get(nid, "?")))
for v in calls.values():
    v.sort()
sample_time = dict(db.execute("SELECT c.id, c.start FROM COMPOSITE_EVENTS c JOIN sel USING (id)"))
api_of = collections.Counter()
for sid, tid, _ in inside:
    v, t = calls.get(tid, []), sample_time[sid]
    i = bisect.bisect_right(v, (t, float("inf"), "")) - 1
    api_of[v[i][2] if i >= 0 and v[i][1] >= t else "(no CUDA call: host code)"] += 1
leafmod = collections.defaultdict(collections.Counter)
for sid, mod, depth in db.execute("SELECT c.id, c.module, c.stackDepth FROM SAMPLING_CALLCHAINS c JOIN sel USING (id) "
                                  "WHERE c.stackDepth = 0"):
    leafmod[sid] = S.get(mod, "?").split("\\")[-1]
per_thread = collections.defaultdict(collections.Counter)
for sid, tid, _ in inside:
    per_thread[tid][leafmod.get(sid, "?")] += 1
print("threads: samples overall / during GPU idle / CUDA API calls made / top leaf modules during idle")
for tid in sorted(total, key=lambda t: -total[t])[:12]:
    print(f"  {tid & 0xFFFFFF:6d}: {total[tid]:6d} / {by_thread[tid]:6d} / {len(calls.get(tid, [])):7d} / "
          + ", ".join(f"{m} {n}" for m, n in per_thread[tid].most_common(3)))
print("CUDA API call the sampled thread was inside (samples during GPU idle):")
for k, n in api_of.most_common(10):
    print(f"  {n:7d}  {100 * n / max(1, len(inside)):5.1f}%  {k}")
leaf, ct2, py = collections.Counter(), collections.Counter(), collections.Counter()
for sid, stack in frames.items():
    leaf[f"{stack[0][0]}!{stack[0][1][:70]}"] += 1
    ct2[next((f[1][:90] for f in stack if "ctranslate2" in f[0].lower()), "(no ctranslate2 frame)")] += 1
    py[next((f"{f[0]}!{f[1][:70]}" for f in stack if "python" in f[0].lower() or f[0].endswith(".py")),
            "(no python frame)")] += 1
for title, c in (("leaf function", leaf), ("innermost ctranslate2 frame", ct2), ("innermost python frame", py)):
    print(f"top {title} (samples during GPU idle):")
    for k, n in c.most_common(14):
        print(f"  {n:7d}  {k}")
if "--stacks" in sys.argv:
    for sid in list(frames)[: int(sys.argv[sys.argv.index("--stacks") + 1])]:
        print("stack:", " <- ".join(f"{m}!{s[:50]}" for m, s in frames[sid][:14]))
