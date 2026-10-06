"""A GPU-metrics capture's averages and percentiles per metric (nsys profile --gpu-metrics-devices, exported to sqlite):
DRAM read and write bandwidth, SM and tensor-core activity, ... (gpumetrics.sh, longform.sh METRICS=).
usage: gpu_metrics_summary.py <capture.sqlite>"""
import collections, sqlite3, sys

db = sqlite3.connect(sys.argv[1])
tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
if "GPU_METRICS" not in tables:
    print("no GPU_METRICS table:", sorted(tables))
    sys.exit()
names = dict(db.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS"))
values = collections.defaultdict(list)
for metric, value in db.execute("SELECT metricId, value FROM GPU_METRICS"):
    values[metric].append(value)
for metric, vs in sorted(values.items()):
    vs.sort()
    n = len(vs)
    print(f"{names.get(metric, metric)[:60]:60} mean {sum(vs) / n:8.1f}  p10 {vs[n // 10]:8.1f}  "
          f"p50 {vs[n // 2]:8.1f}  p90 {vs[9 * n // 10]:8.1f}  (n {n})")
