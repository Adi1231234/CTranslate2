"""Downloads a whisper-bench job's results next to each other for the laptop: results.jsonl, each script line's
out.txt, each configuration's progress logs (<label>.<process>.log) and, with --out, its rows (out/<label>/, the
first pass of a RUN_REPEAT run), then
prints every configuration's steady-state rate: the audio all processes wrote between the moment the last one
logged its first unit and the moment the first one finished, over that span (no model load, no lone tail).
usage: python fetch_results.py <job id> <dest dir> [--out label ...]"""
import argparse, datetime as dt, glob, json, os, re
from settings import BUCKET, S3_PREFIX, client

p = argparse.ArgumentParser(); p.add_argument("job"); p.add_argument("dest"); p.add_argument("--out", nargs="*", default=[])
a = p.parse_args()
s3, prefix = client("s3"), f"{S3_PREFIX}/results/{a.job}/"
os.makedirs(a.dest, exist_ok=True)
keys = [o["Key"] for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix)
        for o in page.get("Contents", [])]
for key in keys:
    rel = key[len(prefix):]
    m = re.match(r"logs/([^/]+)/root(\d+)/progress\.log$", rel)
    if rel == "results.jsonl":
        dest = "results.jsonl"
    elif re.match(r"logs/script[^/]*/out\.txt$", rel):
        dest = rel.split("/")[1] + ".txt"
    elif m:
        dest = f"{m.group(1)}.{m.group(2)}.log"
    elif re.match(r"logs/[^/]+/p\d+\.out$", rel):
        dest = rel.split("/")[1] + "." + rel.split("/")[2]
    elif rel.startswith("out/") and rel.split("/")[1] in a.out and "~" not in rel:   # RUN_REPEAT's passes: not
        dest = rel
    else:
        continue
    path = os.path.join(a.dest, dest)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s3.download_file(BUCKET, key, path)

for line in open(os.path.join(a.dest, "results.jsonl")) if os.path.exists(os.path.join(a.dest, "results.jsonl")) else []:
    r = json.loads(line)
    series = []
    for log in sorted(glob.glob(os.path.join(a.dest, f"{r['label']}.*.log"))):
        pts = [(dt.datetime.strptime(m.group(1), "%H:%M:%S"), float(m.group(2)) * 3600)
               for m in (re.match(r"(\d\d:\d\d:\d\d) .* audio ([\d.]+)h", l) for l in open(log, encoding="utf-8")) if m]
        if pts:
            series.append(pts)
    steady = ""
    if series:
        t0, t1 = max(s[0][0] for s in series), min(s[-1][0] for s in series)
        at = lambda s, t: max((v for when, v in s if when <= t), default=0.0)
        span = (t1 - t0).total_seconds()
        if span > 0:
            steady = f", steady {sum(at(s, t1) - at(s, t0) for s in series) / span:.1f}x over {span:.0f} s"
    print(f"{r['label']}: {r['x_realtime']}x wall ({r['wall_s']} s, {r['audio_h']} h, {r['fallback_rows']} fallback rows,"
          f" GPU busy {r['gpu_busy_pct']}%, mem ctrl {r['gpu_mem_ctrl_pct']}%, peak {r['gpu_mem_peak_mib']} MiB,"
          f" CPU {r['cpu_busy_pct']}%){steady}")
