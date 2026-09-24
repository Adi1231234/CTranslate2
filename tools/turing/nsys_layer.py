"""The kernels of one encoder layer in launch order, from an Nsight Systems report exported to SQLite:
on the stream of the first full-batch encoder call (the conv1d im2col kernel starts it), the kernels
between its 2nd and 4th LayerNorm, with grid, duration and the distinguishing part of the demangled name.
usage: nsys_layer.py <report.sqlite> [encoder call index, default 1]"""
import sys, re, sqlite3

db = sqlite3.connect(sys.argv[1])
call = int(sys.argv[2]) if len(sys.argv) > 2 else 1
S = dict(db.execute("SELECT id, value FROM StringIds"))
starts = db.execute("SELECT start, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id "
                    "WHERE s.value = 'im2col_transposed_kernel' AND gridZ = 8 ORDER BY start").fetchall()
t0, stream = starts[2 * call]                      # two convolutions per encoder call
ks = db.execute("SELECT start, end, shortName, demangledName, gridX, gridY, gridZ, blockX FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "WHERE streamId = ? AND start >= ? ORDER BY start LIMIT 400", (stream, t0)).fetchall()
ln = [i for i, k in enumerate(ks) if S[k[2]] == "LayerNormForwardCUDAKernel"]


def label(name):
    """The informative bit of a long template name: cutlass config, thrust functor or CT2 functor."""
    for pat in (r"cutlass_\w+", r"ctranslate2::(?:\w+::)*(\w+)<", r"thrust::\w+::(\w+)<", r"(\w+)<"):
        m = re.search(pat, name)
        if m:
            return m.group(m.lastindex or 0)[:60]
    return name[:60]


total = 0
for s, e, sn, dn, gx, gy, gz, bx in ks[ln[1]:ln[3]]:
    total += e - s
    print(f"{(e - s) / 1e3:8.1f} us  {gx}x{gy}x{gz}/{bx:<5} {S[sn][:28]:<28} {label(S[dn])}")
print(f"layer total {total / 1e3:.0f} us")
