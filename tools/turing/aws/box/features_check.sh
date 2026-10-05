#!/bin/bash
# ../../scale/features_check.py on the box (an entry.sh "script" line): the chunked features (runner/chunked_features.py)
# against faster-whisper's own on the long sample's two Knesset plenums and two YODAS v3 files, with the box's numpy
# and OpenBLAS, byte for byte. The check comes from S3 next to this script (the image predates it).
# usage: features_check.sh <runner dir>
set -uo pipefail
RUNNER=$1; B=${B:-/opt/wb}; T=$B/src/tools/turing
[ -d "$B/$RUNNER" ] || aws s3 cp --only-show-errors "$S3/$RUNNER.tgz" - | tar xz -C "$B"
aws s3 cp --only-show-errors "$S3/scripts/features_check.py" "$T/scale/features_check.py"
mkdir -p check
for f in knesset-39054.m4a knesset-68632.m4a 0021727bd7ca260b4c4891953683d275.webm 005aeef5f9120c0465764f9ee987169d.webm; do
  aws s3 cp --only-show-errors "$S3/data/longsample/audio/$f" "check/$f"
done
$B/venv/bin/python "$T/scale/features_check.py" "$B/$RUNNER" check 2 2>&1 | grep -vE "[Ww]arning|warn\("
