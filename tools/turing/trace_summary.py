"""Readable slices of a trace_kernels.py TSV: the kernels of one encoder layer and of one decode step's
first layers, in launch order, plus the per-kernel totals of that decode step.
usage: trace_summary.py <trace.tsv> [decode step, default 20] [kernels to list, default 90]"""
import sys, collections
path = sys.argv[1]
step = int(sys.argv[2]) if len(sys.argv) > 2 else 20
count = int(sys.argv[3]) if len(sys.argv) > 3 else 90
rows = [l.rstrip("\n").split("\t") for l in open(path)][1:]
enc = [r for r in rows if r[0] == "enc"]
dec = [r for r in rows if r[0] == "dec"]


def show(rs):
    for r in rs:
        print(f"{float(r[2]):8.1f}us  g={r[3]:<12} b={r[4]:<10} sm={r[5]:<6} {r[6][:110]}")


# Encoder: layers start with a LayerNorm; list layer 1 (between the 2nd and 4th LayerNorm).
ln = [i for i, r in enumerate(enc) if "LayerNorm" in r[6]]
print(f"== encoder: {len(enc)} kernels, layer 1 ({ln[2]}..{ln[4]})")
show(enc[ln[2]:ln[4]])
# Decoder: the vocabulary log-softmax (the only row wider than 2048) closes every step.
ends = [i for i, r in enumerate(dec) if "cunn_SoftMaxForward" in r[6]]
a, b = ends[step - 1] + 1, ends[step] + 1
print(f"== decode step {step}: kernels {b - a}, {sum(float(r[2]) for r in dec[a:b]) / 1e3:.2f} ms GPU")
tot, cnt = collections.Counter(), collections.Counter()
for r in dec[a:b]:
    tot[r[6][:90]] += float(r[2]); cnt[r[6][:90]] += 1
for n, t in tot.most_common(16):
    print(f"   {t:8.1f}us x{cnt[n]:4d}  {n}")
print(f"== decode step {step}: first {count} kernels")
show(dec[a:a + count])
