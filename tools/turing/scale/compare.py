"""Row-by-row comparison of two runs of the production runner (out/<unit>.jsonl), e.g. the stock wheel's
production output against a build's re-run of the same units. A row is equal only if its JSON line is
byte-identical (text, every segment's times, log-prob, no-speech probability, compression ratio, temperature,
the clip's duration and the decoding path). Rows decoded at a sampling temperature (a segment with
temperature > 0: the fallback's ladder after T=0 failed) draw random numbers that no two runs repeat, so they
are listed apart (for seeded.py) instead of counted as differences.
usage: compare.py <ref_dir> <new_dir> [--sampled <list file>]    (the units are the new run's files)"""
import os, sys, json, glob


def sampled(row):
    return any(s["temperature"] > 0 for s in row.get("segments") or [])


def changed(a, b):
    """The fields that differ between two rows, and the first differing segment field."""
    keys = [k for k in a.keys() | b.keys() if a.get(k) != b.get(k)]
    for sa, sb in zip(a.get("segments") or [], b.get("segments") or []):
        seg = [k for k in sa if sa[k] != sb.get(k)]
        if seg:
            return keys + [f"segment.{seg[0]}"]
    return keys


args = sys.argv[1:]
ref_dir, new_dir = args[0], args[1]
listing = args[args.index("--sampled") + 1] if "--sampled" in args else None
n = dict(units=0, rows=0, equal=0, sampled=0, differ=0, audio_h=0.0, batch8=0, fallback=0, fallback_t0=0)
found, notes = [], []
for path in sorted(glob.glob(os.path.join(new_dir, "*.jsonl"))):
    uid = os.path.basename(path)[:-len(".jsonl")]
    ref_path = os.path.join(ref_dir, uid + ".jsonl")
    new = open(path, encoding="utf-8").read().splitlines()
    ref = open(ref_path, encoding="utf-8").read().splitlines() if os.path.exists(ref_path) else None
    if ref is None or len(ref) != len(new):
        notes.append(f"{uid}: " + ("no reference" if ref is None else f"{len(ref)} vs {len(new)} rows"))
        n["differ"] += len(new)
        continue
    n["units"] += 1
    for a, b in zip(ref, new):
        ra, rb = json.loads(a), json.loads(b)
        n["rows"] += 1
        n["audio_h"] += rb["dur_s"] / 3600
        path_kind = rb.get("path", "error")
        n[path_kind] = n.get(path_kind, 0) + 1
        if path_kind == "fallback" and not sampled(rb):
            n["fallback_t0"] += 1
        at_random = sampled(ra) or sampled(rb)
        if at_random:
            n["sampled"] += 1
            found.append(f"{uid} {rb['uuid']}")
        if a == b:
            n["equal"] += 1
        elif at_random:
            n["sampled_differ"] = n.get("sampled_differ", 0) + 1
        else:
            n["differ"] += 1
            notes.append(f"{uid} {rb['uuid']}: {','.join(changed(ra, rb))} | {ra.get('text')!r} -> {rb.get('text')!r}")
if listing:
    open(listing, "w", encoding="utf-8").write("".join(line + "\n" for line in found))
n["audio_h"] = round(n["audio_h"], 2)
print(json.dumps({**n, "verdict": "IDENTICAL" if not n["differ"] else "DIFFERENT"}), flush=True)
for line in notes[:40]:
    print(line, flush=True)
