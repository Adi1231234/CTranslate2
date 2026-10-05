#!/bin/bash
# ../../scale/encoder_bench.py (an entry.sh "script" line): the encoder alone on every clip of the units list, in the
# stream's batches of 8, one process. The bench comes from S3 next to this script (the image may predate it).
# usage: encoder_bench.sh <package> <runner dir> [passes]
set -uo pipefail
PKG=$1; RUNNER=$2; PASSES=${3:-2}; B=${B:-/opt/wb}; T=$B/src/tools/turing
for p in "$PKG" "$RUNNER"; do
  [ -d "$B/$p" ] || aws s3 cp --only-show-errors "$S3/$p.tgz" - | tar xz -C "$B"
done
aws s3 cp --only-show-errors "$S3/scripts/encoder_bench.py" "$T/scale/encoder_bench.py"
cat "$B/$PKG/ctranslate2/BUILD.txt"
(export HF_HOME=$B/hf PYTHONPATH=$B/$PKG
 cd "$B/$RUNNER" && $B/venv/bin/python "$T/scale/encoder_bench.py" "$B/$RUNNER" $B/cache $T/scale/units_real.txt \
   "$PASSES" 2>&1 | grep -E "encoder pass|Error|error|Traceback")
