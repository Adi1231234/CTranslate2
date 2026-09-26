"""The GPU work issued to the legacy default stream in an nsys profile (--trace=cuda): each copy, set and kernel
on it with the runtime call that issued it, the host thread, the copy kind and size. Work there waits for every
blocking stream and holds up every blocking stream behind it, so one small copy can stall the decoder behind the
encoder's queued kernels. The legacy stream is the one of the first CT2 worker's copies that no created stream owns;
nsys reports it as a stream id, found here as the stream of cudaMemcpy calls (default: the id given, else 7).
usage: nsys_null_stream.py <profile.sqlite> [stream id=7] [rows to list=20]"""
import collections, sqlite3, sys

db = sqlite3.connect(sys.argv[1])
NULL = int(sys.argv[2]) if len(sys.argv) > 2 else 7
LIST = int(sys.argv[3]) if len(sys.argv) > 3 else 20
S = dict(db.execute("SELECT id, value FROM StringIds"))
api = {cid: (s, e, tid, S.get(n, "?")) for cid, s, e, tid, n in
       db.execute("SELECT correlationId, start, end, globalTid, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
t0 = db.execute("SELECT min(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
tables = {t for (t,) in db.execute("SELECT name FROM sqlite_master WHERE type = ?", ("table",))}
kinds = dict(db.execute("SELECT id, name FROM ENUM_CUDA_MEMCPY_OPER")) if "ENUM_CUDA_MEMCPY_OPER" in tables else {}
rows = []
for table, extra in (("CUPTI_ACTIVITY_KIND_MEMCPY", "copyKind, bytes"), ("CUPTI_ACTIVITY_KIND_MEMSET", "0, bytes"),
                     ("CUPTI_ACTIVITY_KIND_KERNEL", "0, 0")):
    if table in tables:
        for s, e, cid, kind, size in db.execute(f"SELECT start, end, correlationId, {extra} FROM {table} "
                                                "WHERE streamId = ?", (NULL,)):
            rows.append((s, e, table.split("_")[-1].lower(), kinds.get(kind, kind), size, api.get(cid)))
rows.sort()
calls = collections.Counter((r[5][3] if r[5] else "?", r[2], r[3], r[4]) for r in rows)
print(f"{len(rows)} operations on stream {NULL}")
for (name, what, kind, size), n in calls.most_common(12):
    print(f"  {n:5d}x {what} {kind} {size} bytes, issued by {name}")
for s, e, what, kind, size, call in rows[:LIST]:
    who = f"{call[3]} thread {call[2] & 0xFFFFFF}, call {(call[1] - call[0]) / 1e3:.1f} us" if call else "?"
    print(f"  @{(s - t0) / 1e6:9.1f} ms {what} {kind} {size} bytes, GPU {(e - s) / 1e3:.1f} us, {who}")
