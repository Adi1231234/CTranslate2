#!/bin/bash
# The bit-for-bit checks of a package against the stock wheel (an entry.sh "script" line), each run once with the
# stock wheel and once with the package, then compared byte for byte: ../../scale/logits_check.py (every step's full
# logits of greedy decoding) and ../../scale/seed_vs_stock.py (the seeded draws against stock's). The checks and the
# uuid list come from S3 next to this script; the runner dir must hold stock_context.py.
# usage: verify_checks.sh <package> <runner dir> <uuid list file name> <units for the logits>
set -uo pipefail
PKG=$1; RUNNER=$2; LIST=$3; UNITS=$4; B=${B:-/opt/wb}; T=$B/src/tools/turing
for p in "$PKG" "$RUNNER"; do
  [ -d "$B/$p" ] || aws s3 cp --only-show-errors "$S3/$p.tgz" - | tar xz -C "$B"
done
for f in logits_check.py seed_vs_stock.py; do aws s3 cp --only-show-errors "$S3/scripts/$f" "$T/scale/$f"; done
aws s3 cp --only-show-errors "$S3/scripts/$LIST" uuids.txt
cat "$B/$PKG/ctranslate2/BUILD.txt"
export HF_HOME=$B/hf
run() {   # <stock | package> <check> <check arguments after the units list>
  local who=$1 check=$2 t0=$(date +%s); shift 2
  (if [ "$who" != stock ]; then export PYTHONPATH=$B/$who; fi
   $B/venv/bin/python "$T/scale/$check" "$B/$RUNNER" $B/cache $T/scale/units_real.txt "$@" 2>&1 | tail -2)
  echo "$check $who: $(( $(date +%s) - t0 )) s"
}
same() {  # <name> <stock file> <package file>
  if cmp -s "$2" "$3"; then echo "$1 IDENTICAL ($(stat -c %s "$3") bytes)"; else echo "$1 DIFFERENT"; fi
}
run stock logits_check.py "$UNITS" logits_stock.jsonl
run "$PKG" logits_check.py "$UNITS" logits_fork.jsonl
same "LOGITS" logits_stock.jsonl logits_fork.jsonl
run stock seed_vs_stock.py uuids.txt seeds_stock.json
run "$PKG" seed_vs_stock.py uuids.txt seeds_fork.json
same "SEEDED DRAWS" seeds_stock.json seeds_fork.json
