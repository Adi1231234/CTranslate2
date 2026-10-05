"""Where a full run's fallback ladders ended: every fallback row of a run's rows (out/<label>/*.jsonl) with its
segments' temperatures (the attempt each window kept: 0 is the beam attempt, 0.2 .. 1.0 the sampled ones; 1.0 also
when all six failed), and how many attempts that took at most (round34: 7 of the 15 clips end at T = 1.0).
usage: python ladder_stats.py <rows dir>"""
import collections, glob, json, os, sys

LADDER = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
rows = [json.loads(l) for f in glob.glob(os.path.join(sys.argv[1], "*.jsonl")) for l in open(f, encoding="utf-8")]
fallback = [r for r in rows if r.get("path") == "fallback"]
print(len(rows), "rows,", len(fallback), "fallback")
kept = collections.Counter()
attempts = 0
for r in fallback:
    temps = [s["temperature"] for s in r["segments"]]
    kept.update(temps)
    attempts += sum(LADDER.index(t) + 1 for t in temps)
    print(f"{r['uuid'][-30:]:>30} {r['dur_s']:5.1f} s, {len(temps):2d} segments at {temps}")
print("segments by temperature:", dict(sorted(kept.items())), "- attempts, a window a segment:", attempts)
