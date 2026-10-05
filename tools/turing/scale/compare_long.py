"""Two long-recording runs (runner/longform_run.py rows.jsonl) recording by recording, for the recordings in both.
--strict: every row must be byte-identical (two runs drawing the same random numbers, e.g. stock and the fork, both
MODE=seq with RUN_SEED). Otherwise rows may part where a window's ladder sampled (other random numbers): a row is
equal, or parts at a segment; the parting is explained when a sampled segment (temperature > 0) came at or before it
in either run, unexplained otherwise (then printed: that is a real difference, or a ladder whose sampled attempt was
skipped as silence, which leaves no segment, so read it).
usage: compare_long.py <reference rows.jsonl> <new rows.jsonl> [--strict]"""
import json, sys

ref_path, new_path = sys.argv[1:3]
strict = "--strict" in sys.argv
load = lambda p: {(r["source"], r["id"]): (line, r) for line in open(p, encoding="utf-8") for r in [json.loads(line)]}
ref, new = load(ref_path), load(new_path)
keys = [k for k in new if k in ref]
n = dict(recordings=len(keys), equal=0, explained=0, unexplained=0, audio_h=0.0, segments=0)
for key in keys:
    (la, a), (lb, b) = ref[key], new[key]
    n["audio_h"] += b["dur_s"] / 3600
    n["segments"] += len(b["segments"])
    if la == lb:
        n["equal"] += 1
        continue
    sa, sb = a["segments"], b["segments"]
    i = next((i for i, (x, y) in enumerate(zip(sa, sb)) if x != y), min(len(sa), len(sb)))
    sampled = any(s["temperature"] > 0 for s in sa[:i + 1] + sb[:i + 1])
    if sampled and not strict:
        n["explained"] += 1
        continue
    n["unexplained"] += 1
    show = lambda s: f"{s['start']:.2f}-{s['end']:.2f} T={s['temperature']} {s['text'][:50]!r}" if s else "(none)"
    print(f"{key[0]} {key[1]}: segment {i} of {len(sa)}/{len(sb)}: "
          f"{show(sa[i] if i < len(sa) else None)} | {show(sb[i] if i < len(sb) else None)}")
n["audio_h"] = round(n["audio_h"], 3)
print(json.dumps({**n, "verdict": "IDENTICAL" if not n["unexplained"] else "DIFFERENT"}, ensure_ascii=False))
