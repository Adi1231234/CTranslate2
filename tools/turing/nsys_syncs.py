"""CUDA runtime calls of an nsys profile (--trace=cuda) by name: count, total and mean time in the call, and
for the ones that wait for the GPU (memcpy to the host, synchronize) the kernel that ran just before.
usage: nsys_syncs.py <profile.sqlite> [top N, default 20]"""
import bisect, collections, re, sqlite3, sys

db = sqlite3.connect(sys.argv[1])
top = int(sys.argv[2]) if len(sys.argv) > 2 else 20
names = dict(db.execute("SELECT id, value FROM StringIds"))
calls = db.execute("SELECT start, end, nameId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
kern = db.execute("SELECT end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY end").fetchall()
ends = [k[0] for k in kern]


def short(n):
    n = re.sub(r"^void ", "", names.get(n, str(n)))
    return re.split(r"[<(]", n)[0].split("::")[-1][:40]


total, count = collections.Counter(), collections.Counter()
before = collections.Counter()
for s, e, nid, tid in calls:
    name = names.get(nid, str(nid))
    total[name] += e - s; count[name] += 1
    if re.search(r"Memcpy|Synchronize", name) and e - s > 20e3:     # waited for the GPU
        i = bisect.bisect_right(ends, e) - 1
        before[(name[:28], short(kern[i][1]) if i >= 0 else "-")] += 1
print(f"{'call':40s}{'n':>9s}{'total s':>9s}{'mean us':>9s}")
for name, t in total.most_common(top):
    print(f"{name[:40]:40s}{count[name]:9d}{t / 1e9:9.3f}{t / count[name] / 1e3:9.1f}")
print("== waits over 20 us by the kernel that ended last")
for (name, k), n in before.most_common(top):
    print(f"{n:8d}  {name:28s} after {k}")
