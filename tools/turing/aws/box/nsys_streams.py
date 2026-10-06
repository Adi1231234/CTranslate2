"""Where an Nsight Systems capture's GPU time went, CUDA stream by stream (longform.sh, NSYS=...): each stream's
kernel time and share of the capture, and its heaviest kernels, so the long engine's parts can be told apart (the
joint stream, the encoder, each ladder lane: every CTranslate2 worker has a stream of its own).
usage: nsys_streams.py <capture.sqlite> (nsys export --type sqlite)"""
import sqlite3, sys

db = sqlite3.connect(sys.argv[1])
q = lambda sql, *a: db.execute(sql, a).fetchall()
lo, hi = q("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL")[0]
span = (hi - lo) / 1e9
total = q("SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL")[0][0] / 1e9
print(f"capture {span:.2f} s, kernel time {total:.2f} s over all streams")
for stream, n, t in q("SELECT streamId, COUNT(*), SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL "
                      "GROUP BY streamId ORDER BY 3 DESC"):
    t /= 1e9
    print(f"stream {stream}: {n} kernels, {t:.2f} s = {100 * t / total:.1f}% of the kernel time, "
          f"{100 * t / span:.1f}% of the capture")
    for name, k, kt in q("SELECT s.value, COUNT(*), SUM(k.end - k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                         "JOIN StringIds s ON k.shortName = s.id WHERE k.streamId = ? GROUP BY s.value "
                         "ORDER BY 3 DESC LIMIT 8", stream):
        print(f"    {kt / 1e9:7.3f} s {k:8d}  {name[:110]}")
