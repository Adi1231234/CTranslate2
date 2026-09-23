"""What the host CPU does while the GPU is idle, from an Nsight Systems report with CPU sampling
(--sample=process-tree, --python-sampling=true), exported to SQLite.
usage: nsys_cpu.py <report.sqlite> --schema   (tables and columns of the sampling data)"""
import sys, sqlite3

db = sqlite3.connect(sys.argv[1])
if "--schema" in sys.argv:
    for (t,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        n = db.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        cols = [c[1] for c in db.execute(f"PRAGMA table_info({t})")]
        print(f"{t} ({n} rows): {', '.join(cols)}")
