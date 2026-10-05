#!/bin/bash
# ../../scale/sampled_check.py twice on a package (an entry.sh "script" line): the memory repeated a hypothesis
# (CT2_SHARED_MEMORY_ROWS=0, stock) and shared, same seed, then the two outputs compared byte for byte. The check
# and its uuid list come from S3 next to this script (the image may predate them).
# usage: sampled_check.sh <package> <uuid list file name>
set -uo pipefail
PKG=$1; LIST=$2; B=${B:-/opt/wb}; T=$B/src/tools/turing
[ -d "$B/$PKG" ] || aws s3 cp --only-show-errors "$S3/$PKG.tgz" - | tar xz -C "$B"
aws s3 cp --only-show-errors "$S3/scripts/sampled_check.py" "$T/scale/sampled_check.py"
aws s3 cp --only-show-errors "$S3/scripts/$LIST" uuids.txt
export HF_HOME=$B/hf PYTHONPATH=$B/$PKG
for mode in 0 1; do
  t0=$(date +%s)
  CT2_SHARED_MEMORY_ROWS=$mode $B/venv/bin/python "$T/scale/sampled_check.py" $B/runner $B/cache \
    $T/scale/units_real.txt uuids.txt "shared$mode.json" 2>&1 | tail -2
  echo "CT2_SHARED_MEMORY_ROWS=$mode: $(( $(date +%s) - t0 )) s"
done
if cmp -s shared0.json shared1.json; then echo "IDENTICAL sampled attempts"; else echo "DIFFERENT"; fi
