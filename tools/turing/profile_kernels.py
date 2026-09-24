"""Kernel-level profile of CTranslate2 Whisper on this GPU, from CUPTI activity records.
Phase A: encoder of one 8-clip batch, then its beam-5 decode. Phase B: next batch's encoder on a
second worker while batch 0 decodes. Per phase: wall, GPU-busy (union of kernel intervals), top
kernels, and how long kernels of different streams actually ran at the same time.
usage: profile_kernels.py <sample_dir> <out_log> [ctranslate2 package parent dir]"""
import os, sys, collections, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
SAMPLE, OUT = sys.argv[1], sys.argv[2]
common.init(sys.argv[3] if len(sys.argv) > 3 else None)
from cupti import Tracer, short, busy
tr = Tracer(os.path.join(os.path.dirname(OUT), "cupti"))
m, (f0, f1), gen = common.load(SAMPLE, batches=2, num_workers=2)
gen(m.encode(f0)); gen(m.encode(f1))                            # warmup, outside the trace
tr.start()
t0 = tr.now(); e0 = m.encode(f0); t1 = tr.now(); out = gen(e0); t2 = tr.now()
steps = max(len(r.sequences_ids[0]) for r in out) + 1
th = threading.Thread(target=m.encode, args=(f1,)); t3 = tr.now(); th.start(); gen(e0); th.join(); t4 = tr.now()
tr.stop()
lines = []
for tag, a, b in (("encoder_bs8", t0, t1), ("decode_bs8_beam5", t1, t2), ("enc_next||decode", t3, t4)):
    rs = [r[:4] for r in tr.records if a <= r[2] <= b]
    tot = collections.Counter(); cnt = collections.Counter()
    for n, s, st, en in rs:
        tot[short(n)] += en - st; cnt[short(n)] += 1
    wall, sumk = (b - a) / 1e6, sum(en - st for _, _, st, en in rs) / 1e6
    streams = sorted(set(r[1] for r in rs))
    per = {s: busy([(st, en) for _, ss, st, en in rs if ss == s]) / 1e6 for s in streams}
    allbusy = busy([(st, en) for _, _, st, en in rs]) / 1e6
    lines.append(f"== {tag}: wall {wall:.1f} ms | kernels {len(rs)} | kernel-time {sumk:.1f} ms | "
                 f"GPU busy {allbusy:.1f} ms | per-stream busy "
                 + ", ".join(f"s{s}:{v:.1f}" for s, v in per.items())
                 + f" | streams overlapped {sum(per.values()) - allbusy:.1f} ms"
                 + (f" | decode steps {steps}, kernels/step {len(rs) / steps:.0f}" if tag.startswith("decode") else ""))
    for n, t in tot.most_common(14):
        lines.append(f"   {t / 1e6:8.1f} ms {100 * t / max(1, sum(tot.values())):5.1f}% x{cnt[n]:5d}  {n}")
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
del m, e0, out                                  # release the model's worker threads while Python is alive
import gc; gc.collect()
