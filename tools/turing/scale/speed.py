"""Wall time of a scale run against the production run on the same units, from the two progress logs.
A unit's production time is the gap between its "ok" line and the one before it (production wrote every unit
in order, one run after another); the scale run's time is its own elapsed time from START to its last unit.
usage: speed.py <production progress.log> <scale run progress.log>"""
import re, sys, json
from datetime import datetime, timedelta

LINE = re.compile(r"^(\d\d:\d\d:\d\d) (\S+) ok \| units \d+ \| audio ([\d.]+)h")


def events(path):
    """(time, unit, cumulative audio hours) per written unit, and the START times, with midnight crossings."""
    out, starts, day, last = [], [], datetime(2000, 1, 1), None
    for line in open(path, encoding="utf-8", errors="replace"):
        t = datetime.strptime(line[:8], "%H:%M:%S") if re.match(r"\d\d:\d\d:\d\d", line) else None
        if t is None:
            continue
        t = day.replace(hour=t.hour, minute=t.minute, second=t.second)
        if last and t < last - timedelta(hours=1):
            day += timedelta(days=1); t += timedelta(days=1)
        last = t
        m = LINE.match(line)
        if m:
            out.append((t, m.group(2), float(m.group(3))))
        elif " START " in line:
            starts.append(t)
    return out, starts


prod, _ = events(sys.argv[1])
run, starts = events(sys.argv[2])
gap, audio = {}, {}
for (t0, _, a0), (t1, u, a1) in zip(prod, prod[1:]):
    if a1 > a0:                                   # same production process (a restart resets the counters)
        gap[u], audio[u] = (t1 - t0).total_seconds(), a1 - a0
units = [u for _, u, _ in run if u in gap]            # the first unit after a production restart has no gap
hours = sum(audio[u] for u in units)
prod_s = sum(gap[u] for u in units)
scale_s = (run[-1][0] - starts[-1]).total_seconds()     # the last START: one uninterrupted scale run
print(json.dumps({"production": {"units": len(units), "audio_h": round(hours, 2), "seconds": round(prod_s),
                                 "realtime": round(hours * 3600 / prod_s, 1)},
                  "scale_run": {"units": len(run), "audio_h": run[-1][2], "seconds": round(scale_s),
                                "realtime": round(run[-1][2] * 3600 / scale_s, 1)}}))
