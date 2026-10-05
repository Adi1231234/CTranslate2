#!/bin/bash
# ../../scale/sampled_check.py's joined check alone (an entry.sh "script" line): the fallback clips joined in one
# call (group_size=1) against each alone, the beam and temperature-0 sampling, then sampling with each clip's own
# seed (sampling_seeds) at three temperatures, joined against alone and alone repeated. The check and its uuid list
# come from S3 next to this script (the image may predate them).
# usage: joined_check.sh <package> <uuid list file name>
set -uo pipefail
PKG=$1; LIST=$2; B=${B:-/opt/wb}; T=$B/src/tools/turing
[ -d "$B/$PKG" ] || aws s3 cp --only-show-errors "$S3/$PKG.tgz" - | tar xz -C "$B"
aws s3 cp --only-show-errors "$S3/scripts/sampled_check.py" "$T/scale/sampled_check.py"
aws s3 cp --only-show-errors "$S3/scripts/$LIST" uuids.txt
export HF_HOME=$B/hf PYTHONPATH=$B/$PKG
cat "$B/$PKG/ctranslate2/BUILD.txt"
CHECK=joined $B/venv/bin/python "$T/scale/sampled_check.py" $B/runner $B/cache $T/scale/units_real.txt uuids.txt \
  joined.json 2>&1 | grep -E "identical|clips ->|Error|error"
