#!/bin/bash
# ../../scale/ladder_bench.py (an entry.sh "script" line): the fallback clips' ladders alone through the runner's
# pool, once per setting, each a fresh process. A setting is VAR=value[,VAR=value...] (the RUN_FALLBACK* variables).
# The bench and its uuid list come from S3 next to this script (the image may predate them).
# usage: ladder_bench.sh <package> <runner dir> <uuid list file name> <setting> [setting ...]
set -uo pipefail
PKG=$1; RUNNER=$2; LIST=$3; shift 3; B=${B:-/opt/wb}; T=$B/src/tools/turing
for p in "$PKG" "$RUNNER"; do
  [ -d "$B/$p" ] || aws s3 cp --only-show-errors "$S3/$p.tgz" - | tar xz -C "$B"
done
aws s3 cp --only-show-errors "$S3/scripts/ladder_bench.py" "$T/scale/ladder_bench.py"
aws s3 cp --only-show-errors "$S3/scripts/$LIST" uuids.txt; U=$(pwd)/uuids.txt
cat "$B/$PKG/ctranslate2/BUILD.txt"
for setting in "$@"; do
  echo "=== $(date +%T) ${setting//,/ }"
  (export HF_HOME=$B/hf PYTHONPATH=$B/$PKG; for kv in ${setting//,/ }; do export "$kv"; done
   cd "$B/$RUNNER" && $B/venv/bin/python "$T/scale/ladder_bench.py" "$B/$RUNNER" $B/cache $T/scale/units_real.txt \
     "$U" 2>&1 | grep -E "ladders:|Error|error|Traceback" )
done
