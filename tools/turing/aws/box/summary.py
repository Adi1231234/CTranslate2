"""One JSON line for a run.sh configuration: units and rows written, audio hours, wall seconds, x realtime, the
fallback rows, GPU busy % (nvidia-smi: some kernel running), memory-controller %, power and peak memory, and the
host CPU in use (vmstat). Also appended to /opt/wb/results.jsonl.
usage: summary.py <label> <out dir> <log dir> <t0> <t1> <failed processes>"""
import glob, json, os, statistics, sys

label, out, log, t0, t1, failed = sys.argv[1:7]
rows = [json.loads(l) for f in glob.glob(os.path.join(out, "*.jsonl")) for l in open(f, encoding="utf-8")]
audio = sum(r.get("dur_s") or 0 for r in rows)
wall = float(t1) - float(t0)
gpu = [[float(x) for x in l.split(",")] for l in open(os.path.join(log, "gpu.csv")) if l.count(",") == 4]
cpu = [l.split() for l in open(os.path.join(log, "cpu.txt")) if l.split() and l.split()[0].isdigit()]
mean = lambda xs: round(statistics.mean(xs), 1) if xs else None
res = {"label": label, "units": len(glob.glob(os.path.join(out, "*.jsonl"))), "rows": len(rows),
       "audio_h": round(audio / 3600, 3), "wall_s": round(wall, 1), "x_realtime": round(audio / wall, 2),
       "fallback_rows": sum(r.get("path") == "fallback" for r in rows), "failed_processes": int(failed),
       "gpu_busy_pct": mean([g[0] for g in gpu]), "gpu_mem_ctrl_pct": mean([g[1] for g in gpu]),
       "gpu_mem_peak_mib": max((g[2] for g in gpu), default=None), "gpu_power_w": mean([g[3] for g in gpu]),
       "cpu_busy_pct": mean([100 - float(c[14]) for c in cpu])}           # vmstat column 15 = idle
print(json.dumps(res), flush=True)
with open("/opt/wb/results.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
