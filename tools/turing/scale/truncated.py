"""List the batch rows of runner output folders that stop over 1 s before their clip's end: batched decoding can
end a clip before the audio does (crowd-v5, 27.9: 4,379 such rows, where the sequential path transcribes to the
end). Prints seeded.py's "unit uuid" lines, so those clips can be re-run on the sequential path.
usage: truncated.py <runner output dir>... > list.txt        (a folder holds <unit>.jsonl files)"""
import os, sys, glob, json

GAP_S = 1.0
sys.stdout.reconfigure(encoding="utf-8", newline="\n")
n = rows = 0
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        unit = os.path.basename(f)[:-6]
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            rows += 1
            if r["path"] == "batch8" and r["segments"] and r["dur_s"] - r["segments"][-1]["end"] > GAP_S:
                print(unit, r["uuid"]); n += 1
print(f"{n} of {rows} rows end over {GAP_S} s before their clip", file=sys.stderr)
