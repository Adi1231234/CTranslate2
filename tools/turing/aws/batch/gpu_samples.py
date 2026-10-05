"""How a run spent its time against the power limit: histograms of a configuration's GPU samples (<label>.gpu.csv,
nvidia-smi every 0.5 s: utilization, memory controller, memory, power, SM clock; fetch_results.py downloads them)
while busy, power by 10 W and SM clock by 100 MHz, and the samples at the limit (>= 340 W) against those below 320 W
with their clocks (round33: 31-54% at the limit at ~2100 MHz, 10-18% below 320 W near 2350 MHz, 8-14% at 2520).
usage: python gpu_samples.py <label>.gpu.csv [...]"""
import collections, csv, sys

for path in sys.argv[1:]:
    rows = [list(map(float, r)) for r in csv.reader(open(path)) if len(r) == 5]
    busy = [r for r in rows if r[0] > 50]
    if not busy:
        print(path, "no busy samples")
        continue
    n = len(busy)
    mean = lambda xs, i: sum(x[i] for x in xs) / max(len(xs), 1)
    power = collections.Counter(min(int(r[3] // 10) * 10, 350) for r in busy)
    clock = collections.Counter(int(r[4] // 100) * 100 for r in busy)
    cap, low = [r for r in busy if r[3] >= 340], [r for r in busy if r[3] < 320]
    print(f"{path}: {len(rows)} samples, {n} busy")
    print("  power:", ", ".join(f"{k}: {v * 100 / n:.0f}%" for k, v in sorted(power.items())))
    print("  SM clock:", ", ".join(f"{k}: {v * 100 / n:.0f}%" for k, v in sorted(clock.items())))
    print(f"  >= 340 W: {len(cap) * 100 / n:.0f}% at {mean(cap, 4):.0f} MHz; < 320 W: {len(low) * 100 / n:.0f}% at "
          f"{mean(low, 4):.0f} MHz, memory controller {mean(low, 1):.0f}%")
