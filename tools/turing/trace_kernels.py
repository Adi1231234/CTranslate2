"""Every kernel of one encoder batch and its beam-5 decode, in launch order, as TSV: which kernel runs
which op (shape via grid/block), to pick what to replace.
usage: trace_kernels.py <sample_dir> <out_tsv> [ctranslate2 package parent dir]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
SAMPLE, OUT = sys.argv[1], sys.argv[2]
common.init(sys.argv[3] if len(sys.argv) > 3 else None)
from cupti import Tracer, short
tr = Tracer(os.path.join(os.path.dirname(OUT), "cupti"))
m, (f0,), gen = common.load(SAMPLE, batches=1)
gen(m.encode(f0))                                               # warmup, outside the trace
tr.start()
t0 = tr.now(); e0 = m.encode(f0); t1 = tr.now(); out = gen(e0)
tr.stop()
with open(OUT, "w") as f:
    f.write("phase\tstart_us\tdur_us\tgrid\tblock\tsmem\tkernel\n")
    for n, s, st, en, g, b, sm in sorted(tr.records, key=lambda r: r[2]):
        f.write(f"{'enc' if st < t1 else 'dec'}\t{(st - t0) / 1e3:.1f}\t{(en - st) / 1e3:.1f}\t"
                f"{g[0]}x{g[1]}x{g[2]}\t{b[0]}x{b[1]}x{b[2]}\t{sm}\t{short(n, 140)}\n")
print("steps", max(len(r.sequences_ids[0]) for r in out) + 1, "records", len(tr.records), flush=True)
del m, e0, out                                  # release the model's worker threads while Python is alive
import gc; gc.collect()
