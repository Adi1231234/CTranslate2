"""The decoder's long gaps in a pipelined profile next to the host's NVTX ranges (nsys --trace=cuda,nvtx with the
library's ranges: submit encode/generate, encode, generate, prompt, decode): for each decoder-stream gap of at
least MIN ms, every range of every thread that overlaps it, from when to when relative to the gap start.
usage: nsys_nvtx_gaps.py <profile.sqlite> [min_gap_ms=20] [gaps to list=10]"""
import collections, sqlite3, sys

db = sqlite3.connect(sys.argv[1])
MIN = float(sys.argv[2] if len(sys.argv) > 2 else 20) * 1e6
LIST = int(sys.argv[3]) if len(sys.argv) > 3 else 10
S = dict(db.execute("SELECT id, value FROM StringIds"))
kern = sorted(db.execute("SELECT start, end, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL"))
times = collections.defaultdict(list)
for s, e, st in kern:
    times[st].append(e - s)
decoder = {st for st, d in times.items() if sum(d) / len(d) <= 100_000}
gaps, end = [], None
for s, e, st in kern:
    if st not in decoder:
        continue
    if end is not None and s - end >= MIN:
        gaps.append((end, s))
    end = e if end is None else max(end, e)
ranges = [(s, e, tid, text or S.get(tid_text, "?")) for s, e, tid, text, tid_text in db.execute(
    "SELECT start, end, globalTid, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL")]
t0 = kern[0][0]
print(f"{len(gaps)} decoder gaps >= {MIN / 1e6:g} ms, {sum(b - a for a, b in gaps) / 1e6:.0f} ms; {len(ranges)} ranges")
for a, b in gaps[:LIST]:
    print(f"\n@{(a - t0) / 1e6:9.1f} ms gap {(b - a) / 1e6:6.1f} ms")
    for s, e, tid, text in sorted(r for r in ranges if r[1] > a and r[0] < b):
        print(f"   thread {tid & 0xFFFFFF:6d} {text:16s} {(s - a) / 1e6:+9.1f} .. {(e - a) / 1e6:+9.1f} ms")
