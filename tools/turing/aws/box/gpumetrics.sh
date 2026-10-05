#!/bin/bash
# Device-wide hardware counters while a production run goes (an entry.sh "script" line): run.sh starts in the
# background, and after a warm-up Nsight Systems samples the GPU's metrics (DRAM bandwidth, SM and tensor-core
# activity; all processes, MPS included) for a fixed window; the table is summarized as averages and percentiles.
# usage: gpumetrics.sh <label> <package> <runner dir> <units list> <processes> <warm-up s> <window s> [VAR=value ...]
set -uo pipefail
LABEL=$1 PKG=$2 RUNNER=$3 LIST=$4 N=$5 WARM=$6 WIN=$7; shift 7
B=${B:-/opt/wb}
[ -d "$B/$PKG" ] || { aws s3 cp --only-show-errors "$S3/$PKG.tgz" - | tar xz -C "$B"; }
bash "$B/src/tools/turing/aws/box/run.sh" "$LABEL" "$PKG" "$RUNNER" "$LIST" "$N" "$@" > run.out 2>&1 &
RUN=$!
sleep "$WARM"
nsys profile --trace=none --sample=none --cpuctxsw=none --gpu-metrics-devices=all --gpu-metrics-frequency=2000 \
  --duration="$WIN" --force-overwrite=true -o metrics sleep "$((WIN + 5))" > nsys.out 2>&1
tail -3 nsys.out
nsys export --type=sqlite --force-overwrite=true -o metrics.sqlite metrics.nsys-rep > /dev/null 2>&1
$B/venv/bin/python - metrics.sqlite <<'EOF'
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
tabs = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
if "GPU_METRICS" not in tabs:
    print("no GPU_METRICS table:", sorted(tabs)); sys.exit()
names = dict(db.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS"))
vals = collections.defaultdict(list)
for mid, v in db.execute("SELECT metricId, value FROM GPU_METRICS"):
    vals[mid].append(v)
for mid, vs in sorted(vals.items()):
    vs.sort(); n = len(vs)
    print(f"{names.get(mid, mid)[:60]:60} mean {sum(vs)/n:8.1f}  p10 {vs[n//10]:8.1f}  p50 {vs[n//2]:8.1f}  p90 {vs[9*n//10]:8.1f}  (n {n})")
EOF
wait $RUN
tail -2 run.out
