"""Decode steps vs the rest, from an Nsight Systems report exported to SQLite. A decode step ends with
the vocabulary-wide softmax (the only rows too wide for warp_softmax_forward, cunn_SoftMaxForward), so
on each stream the kernels between two of them are one step. Prints the steps' totals, the per-step
kernel mix (name + grid, by GPU time), one median step in launch order, and the kernels outside steps
(encoder, first steps, fallback).
usage: nsys_steps.py <report.sqlite> [kernels of the listed step, default 120] [--grids]
--grids: the per-step mix split by launch grid too (e.g. one GEMM kernel's different shapes).
--pick=<q>: list the step at quantile q of GPU time instead of the median (0.95: a full batch)."""
import sys, sqlite3, collections, statistics

db = sqlite3.connect(sys.argv[1])
args = [a for a in sys.argv[2:] if not a.startswith("--")]
pick = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--pick=")), 0.5))
show = int(args[0]) if args else 120
by_grid = "--grids" in sys.argv
S = dict(db.execute("SELECT id, value FROM StringIds"))
rows = db.execute("SELECT start, end, streamId, shortName, gridX, gridY, gridZ, blockX FROM CUPTI_ACTIVITY_KIND_KERNEL"
                  " ORDER BY start").fetchall()
streams = collections.defaultdict(list)
for s, e, st, n, gx, gy, gz, bx in rows:
    streams[st].append((s, e, S.get(n, "?"), f"{gx}x{gy}x{gz}/{bx}"))
steps, rest = [], []
for ks in streams.values():
    cur = []
    for k in ks:
        cur.append(k)
        if k[2] == "cunn_SoftMaxForward":
            steps.append(cur); cur = []
    rest += cur
first = [st for st in steps if any(k[2] == "im2col_transposed_kernel" for k in st) or st[-1][1] - st[0][0] > 50e6]
steps = [st for st in steps if st not in first]
rest += [k for st in first for k in st]
gpu = [sum(e - s for s, e, _, _ in st) for st in steps]
span = [st[-1][1] - st[0][0] for st in steps]
print(f"{len(steps)} decode steps: GPU {sum(gpu) / 1e6:.0f} ms, span {sum(span) / 1e6:.0f} ms, "
      f"kernels/step {statistics.mean(len(st) for st in steps):.0f}, GPU/step {statistics.mean(gpu) / 1e3:.0f} us, "
      f"span/step {statistics.mean(span) / 1e3:.0f} us; outside steps: {len(rest)} kernels, "
      f"GPU {sum(e - s for s, e, _, _ in rest) / 1e6:.0f} ms")
tot, cnt = collections.Counter(), collections.Counter()
for st in steps:
    for s, e, n, g in st:
        key = f"{n[:48]} {g}" if by_grid else n[:60]
        tot[key] += e - s; cnt[key] += 1
print("\nper-step kernel mix (ms total, calls per step, us per call):")
for n, t in tot.most_common(40 if by_grid else 25):
    print(f"  {t / 1e6:8.1f} ms  {cnt[n] / len(steps):6.1f}/step  {t / cnt[n] / 1e3:7.1f} us  {n}")
mid = sorted(range(len(steps)), key=lambda i: gpu[i])[min(len(steps) - 1, int(pick * len(steps)))]
st = steps[mid]
print(f"\nmedian step: {len(st)} kernels, GPU {gpu[mid] / 1e3:.0f} us, span {span[mid] / 1e3:.0f} us")
for s, e, n, g in st[:show]:
    print(f"  {(s - st[0][0]) / 1e3:8.1f} +{(e - s) / 1e3:6.1f} us  {g:<22} {n[:70]}")
tot, cnt = collections.Counter(), collections.Counter()
for s, e, n, g in rest:
    tot[n[:60]] += e - s; cnt[n[:60]] += 1
print("\noutside decode steps:")
for n, t in tot.most_common(15):
    print(f"  {t / 1e6:8.1f} ms  x{cnt[n]:6d}  {t / cnt[n] / 1e3:8.1f} us  {n}")
