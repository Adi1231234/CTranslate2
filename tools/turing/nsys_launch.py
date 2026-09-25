"""Who makes the GPU wait between kernels: each idle gap before a kernel (GPU-wide, all streams) is put down to
the host when the kernel's launch call returned after the previous kernel had already ended (the GPU had
nothing queued), else to the GPU (the kernel was queued in time; the gap is launch latency). Needs an nsys
profile with --trace=cuda (runtime API records).
usage: nsys_launch.py <profile.sqlite>"""
import collections, sqlite3, sys

db = sqlite3.connect(sys.argv[1])
api = dict(db.execute("SELECT correlationId, end FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
rows = db.execute("SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
buckets = [(3e3, "<3 us"), (10e3, "3-10 us"), (100e3, "10-100 us"), (1e6, "0.1-1 ms"), (float("inf"), ">=1 ms")]
host, gpu = collections.Counter(), collections.Counter()
nh, ng = collections.Counter(), collections.Counter()
lead = []                                            # how far ahead of the GPU the host launched (queue depth)
end = None
for s, e, cid in rows:
    launched = api.get(cid)
    if end is not None and s > end and launched is not None:
        gap = s - end
        b = next(name for top, name in buckets if gap < top)
        if launched > end:
            host[b] += gap; nh[b] += 1
        else:
            gpu[b] += gap; ng[b] += 1
    if launched is not None:
        lead.append(s - launched)
    end = e if end is None else max(end, e)
print(f"kernels {len(rows)}, with a launch record {len(lead)}")
print(f"{'gap length':12s}{'host s':>9s}{'n':>9s}{'gpu s':>9s}{'n':>9s}")
for _, b in buckets:
    print(f"{b:12s}{host[b] / 1e9:9.3f}{nh[b]:9d}{gpu[b] / 1e9:9.3f}{ng[b]:9d}")
print(f"{'all':12s}{sum(host.values()) / 1e9:9.3f}{sum(nh.values()):9d}{sum(gpu.values()) / 1e9:9.3f}{sum(ng.values()):9d}")
lead.sort()
q = lambda f: lead[int(f * (len(lead) - 1))] / 1e3
print(f"launch to start (us): p10 {q(0.1):.1f} p50 {q(0.5):.1f} p90 {q(0.9):.1f} p99 {q(0.99):.1f}")
