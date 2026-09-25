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
rows = db.execute("SELECT start, end, streamId, demangledName, gridX, gridY, gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL "
                  "ORDER BY start").fetchall()
by_stream = collections.defaultdict(list)
for r in rows:
    by_stream[r[2]].append(r[1] - r[0])
enc_streams = {s for s, d in by_stream.items() if sum(d) / len(d) > 100e3}


def family(name):                                    # e.g. gemm_64x64_32x6_tn_align8, exact_attention_kernel
    m = re.search(r"cutlass_80_(?:wmma_)?tensorop_f16_s\d+gemm_(?:relu_)?f16_(\w+)", name)
    if m:
        return ("wmma_" if "wmma" in name else "gemm_") + m.group(1)
    return re.split(r"[<(]", re.sub(r"^void ", "", name))[0].split("::")[-1][:40]


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
    charged.append((key, e - s, gap, fam))
median = {k: (statistics.median(d for d, _ in v), statistics.median(g for _, g in v)) for k, v in alone.items()}
actual, expected, count = collections.Counter(), collections.Counter(), collections.Counter()
extra_dur, extra_gap, by_kernel = collections.Counter(), collections.Counter(), collections.Counter()
for key, dur, gap, fam in charged:
    if key in median:
        actual[fam] += dur + gap; expected[fam] += sum(median[key]); count[fam] += 1
        extra_dur[fam] += dur - median[key][0]; extra_gap[fam] += gap - median[key][1]
        by_kernel[(fam, family(names[key[0]]), key[1:])] += dur + gap - sum(median[key])
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
print("== the extra time as longer kernels vs longer gaps before them")
for fam in sorted(actual, key=lambda f: -(actual[f] - expected[f]))[:top]:
    print(f"{fam:42s} kernels {extra_dur[fam] / 1e9:7.3f} s  gaps {extra_gap[fam] / 1e9:7.3f} s")
print("== decoder kernels with the most extra time")
for (fam, n, grid), v in by_kernel.most_common(2 * top):
    print(f"{v / 1e9:7.3f} s  beside {fam[:24]:24s} {n} {grid}")
