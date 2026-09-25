"""How much the encoder slows the decoder in a pipelined nsys profile, by encoder kernel. Each decoder kernel is
charged with its duration plus the idle time since the previous decoder kernel ended (launch and dispatch
latency), and put under the encoder kernel family running when it started (or "none"). The expected time is
what the same decoder kernels take with no encoder kernel running (median per kernel name and grid).
A stream is the encoder's when its kernels average over 100 us.
usage: nsys_interfere.py <profile.sqlite> [top N families, default 12]"""
import bisect, collections, re, sqlite3, statistics, sys

db = sqlite3.connect(sys.argv[1])
top = int(sys.argv[2]) if len(sys.argv) > 2 else 12
names = dict(db.execute("SELECT id, value FROM StringIds"))
rows = db.execute("SELECT start, end, streamId, shortName, gridX, gridY, gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL "
                  "ORDER BY start").fetchall()
by_stream = collections.defaultdict(list)
for r in rows:
    by_stream[r[2]].append(r[1] - r[0])
enc_streams = {s for s, d in by_stream.items() if sum(d) / len(d) > 100e3}


def family(name):
    name = re.sub(r"cutlass_80_(wmma_)?tensorop_f16_s1\d+gemm_(relu_)?f16_", "gemm_", name)
    return name[:40]


enc = [(r[0], r[1], family(names[r[3]])) for r in rows if r[2] in enc_streams]
enc_starts = [e[0] for e in enc]


def encoder_at(t):                                  # one encoder runs at a time: its latest kernel, or the one before
    i = bisect.bisect_right(enc_starts, t) - 1
    for j in (i, i - 1):
        if j >= 0 and enc[j][1] >= t:
            return enc[j][2]
    return "none"


dec = [r for r in rows if r[2] not in enc_streams]
alone = collections.defaultdict(list)
charged = []                                         # (key, cost incl. gap, duration, family)
prev_end = {}
for s, e, st, n, gx, gy, gz in dec:
    key = (n, gx, gy, gz)
    gap = s - prev_end[st] if st in prev_end and s - prev_end[st] < 1e6 else 0   # host waits excluded
    prev_end[st] = e
    fam = encoder_at(s)
    if fam == "none":
        alone[key].append((e - s, gap))
    charged.append((key, e - s + gap, fam))
median = {k: (statistics.median(d for d, _ in v), statistics.median(g for _, g in v)) for k, v in alone.items()}
actual, expected, count = collections.Counter(), collections.Counter(), collections.Counter()
for key, cost, fam in charged:
    if key in median:
        actual[fam] += cost; expected[fam] += sum(median[key]); count[fam] += 1
enc_time = collections.Counter()
for s, e, f in enc:
    enc_time[f] += e - s
print(f"decoder kernels {len(dec)}, encoder streams {sorted(enc_streams)}")
print(f"{'encoder family running':42s}{'dec n':>9s}{'actual s':>10s}{'alone s':>9s}{'extra s':>9s}{'enc s':>8s}")
for fam, a in sorted(actual.items(), key=lambda x: -(x[1] - expected[x[0]]))[:top]:
    print(f"{fam:42s}{count[fam]:9d}{a / 1e9:10.3f}{expected[fam] / 1e9:9.3f}{(a - expected[fam]) / 1e9:9.3f}"
          f"{enc_time[fam] / 1e9:8.3f}")
tot_a, tot_e = sum(actual.values()), sum(expected.values())
print(f"{'all':42s}{sum(count.values()):9d}{tot_a / 1e9:10.3f}{tot_e / 1e9:9.3f}{(tot_a - tot_e) / 1e9:9.3f}")
