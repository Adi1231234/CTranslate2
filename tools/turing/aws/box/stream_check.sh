#!/bin/bash
# ../../scale/stream_check.py on a package (an entry.sh "script" line): the Whisper stream against generate() on
# each batch alone, bit for bit, and both timed. The check comes from S3 next to this script (the image may
# predate it).
# usage: stream_check.sh <package> [max clips] [max_batches:max_rows ...]
set -uo pipefail
PKG=$1; shift; B=${B:-/opt/wb}; T=$B/src/tools/turing
[ -d "$B/$PKG" ] || aws s3 cp --only-show-errors "$S3/$PKG.tgz" - | tar xz -C "$B"
aws s3 cp --only-show-errors "$S3/scripts/stream_check.py" "$T/scale/stream_check.py"
export HF_HOME=$B/hf PYTHONPATH=$B/$PKG
cat "$B/$PKG/ctranslate2/BUILD.txt"
$B/venv/bin/python "$T/scale/stream_check.py" $B/runner $B/cache $T/scale/units_real.txt stream.json "$@" 2>&1 \
  | grep -v "^\s*$" | tail -12
