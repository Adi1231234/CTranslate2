#!/bin/bash
# Nsight Compute on one kernel probe (an entry.sh "script" line, privileged job): the launches of the kernels matching
# a regex (up to 64) with the instruction statistics and the source counters, then for the launch with the largest
# grid its executed instructions by SASS opcode and its 40 most executed SASS lines (sass.csv: every line, kept in
# the job's logs). Build the probe with -lineinfo (GENCODE in ../../linux/build_probes.sh) to see source lines too.
# usage: ncu_probe.sh <key of probes.tgz under $S3> <probe> <kernel regex> [probe args ...]
set -uo pipefail
aws s3 cp --only-show-errors "$S3/$1" probes.tgz && tar xzf probes.tgz
PROBE=$2; KREGEX=$3; shift 3
NCU=$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V | tail -1)
echo "probes from $(cat probes/BUILD.txt) on $(nvidia-smi --query-gpu=name --format=csv,noheader), $NCU"
"$NCU" -k "regex:$KREGEX" -c 64 --section LaunchStats --section InstructionStats --section SourceCounters \
  --import-source yes -o rep -f "./probes/$PROBE" "$@" > probe.out 2>&1
tail -3 probe.out
"$NCU" --import rep.ncu-rep --page raw --csv --metrics launch__grid_size,smsp__inst_executed.sum,gpu__time_duration.sum \
  > raw.csv
BEST=$(python3 - <<'EOF'
import csv
rows = list(csv.reader(open("raw.csv")))
head, data = rows[0], rows[2:]
num = lambda s: float(s.replace(",", "") or 0)
g, i, t = (head.index(c) for c in ("launch__grid_size", "smsp__inst_executed.sum", "gpu__time_duration.sum"))
best = max(range(len(data)), key=lambda r: num(data[r][g]))
import sys
print(f"launch {best} of {len(data)}: grid {data[best][g]}, {data[best][i]} warp instructions, {data[best][t]} ns",
      file=sys.stderr)
print(best)
EOF
)
"$NCU" --import rep.ncu-rep --launch-skip "$BEST" --launch-count 1 --page source --csv --print-source sass > sass.csv
"$NCU" --import rep.ncu-rep --launch-skip "$BEST" --launch-count 1 --page details --section InstructionStats \
  | grep -vE "^\s*$" | head -40
python3 - <<'EOF'
import csv, collections
rows = list(csv.reader(open("sass.csv")))
head = next(r for r in rows if any("Source" == c or c.startswith("Source") for c in r))
start = rows.index(head) + 1
src = next(k for k, c in enumerate(head) if c.startswith("Source"))
exe = next(k for k, c in enumerate(head) if c.startswith("Instructions Executed"))
lines = []
for r in rows[start:]:
    if len(r) <= max(src, exe):
        continue
    try:
        n = float(r[exe].replace(",", ""))
    except ValueError:
        continue
    lines.append((n, r[src].strip()))
total = sum(n for n, _ in lines)
ops = collections.Counter()
for n, s in lines:
    op = s.split()[0] if s else "?"
    if op.startswith("@"):
        op = s.split()[1] if len(s.split()) > 1 else op
    ops[op.split(".")[0]] += n
print(f"{total:.0f} warp instructions executed")
for op, n in ops.most_common(30):
    print(f"  {op:10s} {n:14.0f} {100 * n / total:5.1f}%")
print("most executed lines:")
for n, s in sorted(lines, reverse=True)[:40]:
    print(f"  {n:12.0f}  {s}")
EOF
