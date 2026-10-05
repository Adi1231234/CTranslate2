"""DRAM traffic by kernel kind, from an Nsight Systems report with CUDA traced and GPU metrics sampled
(aws/box/profile_metrics.sh), exported to SQLite. Every sample's DRAM read and write throughput (percent of the
device's peak over the sample interval) is a sum over the kernels then running; a non-negative least-squares fit of
the samples on the time each kernel kind ran inside them gives each kind's mean bandwidth while it runs, and with
its total running time its share of the traffic. Kinds: kernel names grouped as below, memcpy by direction.
usage: nsys_dram.py <report.sqlite> [peak GB/s, default 864 (L40S)]"""
import collections, sqlite3, sys
import numpy as np

KINDS = [  # (kind, substrings of the demangled name), first match wins; long GEMMs are the encoder's (below)
    ("cross_attention", ["cross_attention_kernel"]),
    ("self_cache_copy", ["reorder_append_kernel"]),
    ("self_attention", ["gemvx::kernel", "gemv2N_kernel", "gemv2T_kernel", "gemvNSP_kernel", "16x16_64x1"]),
    ("encoder_attention", ["exact_attention"]),
    ("norm", ["residual_norm_kernel", "LayerNormForward"]),
    ("softmax", ["softmax_rows1024", "warp_softmax_forward", "cunn_SoftMaxForward"]),
    ("topk", ["topk_stage"]),
    ("conv", ["im2col", "bias_add_block"]),
    ("gemm", ["gemm", "Kernel2", "splitKreduce"]),
]


def kind_of(name, duration_ns):
    for kind, keys in KINDS:
        if any(k in name for k in keys):
            if kind == "gemm":
                return "encoder_gemm" if duration_ns > 150_000 else "decoder_gemm"
            return kind
    return "other_kernels"


def main():
    db = sqlite3.connect(sys.argv[1])
    peak = float(sys.argv[2]) if len(sys.argv) > 2 else 864.0
    S = dict(db.execute("SELECT id, value FROM StringIds"))
    names = dict(db.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS"))
    want = {m: n for m, n in names.items() if n.startswith("DRAM Read Bandwidth") or n.startswith("DRAM Write Bandwidth")}
    samples = collections.defaultdict(dict)
    for ts, mid, v in db.execute("SELECT timestamp, metricId, value FROM GPU_METRICS"):
        if mid in want:
            samples[ts]["read" if "Read" in want[mid] else "write"] = v
    ts = np.array(sorted(samples))
    dt = np.median(np.diff(ts))
    y = {d: np.array([samples[t].get(d, 0.0) for t in ts]) / 100.0 for d in ("read", "write")}
    spans = []
    for s, e, n in db.execute("SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        spans.append((s, e, kind_of(S[n], e - s)))
    for s, e, k in db.execute("SELECT start, end, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
        spans.append((s, e, {1: "memcpy_to_gpu", 2: "memcpy_to_host", 8: "memcpy_on_gpu"}.get(k, "memcpy_other")))
    kinds = sorted({k for _, _, k in spans})
    col = {k: i for i, k in enumerate(kinds)}
    X = np.zeros((len(ts), len(kinds)))
    t_first, t_last = ts[0] - dt, ts[-1]
    total = collections.Counter()
    for s, e, k in spans:                                  # sample i covers (ts[i] - dt, ts[i]]
        if e <= t_first or s >= t_last:
            continue
        s, e = max(s, t_first), min(e, t_last)
        total[k] += e - s
        i = int(np.searchsorted(ts, s, side="left"))
        while i < len(ts) and ts[i] - dt < e:
            X[i, col[k]] += min(e, ts[i]) - max(s, ts[i] - dt)
            i += 1
    X /= dt                                                # fraction of the interval each kind ran
    window = (t_last - t_first) / 1e9
    print(f"window {window:.1f} s, {len(ts)} samples of {dt / 1e3:.0f} us; DRAM read {y['read'].mean() * 100:.1f}%"
          f", write {y['write'].mean() * 100:.1f}% of {peak:.0f} GB/s")
    fit = {}
    for d in ("read", "write"):
        coef = nnls(X, y[d])
        fit[d] = coef
        resid = y[d] - X @ coef
        print(f"  {d} fit: residual mean {resid.mean() * 100:+.1f} points, rms {np.sqrt((resid ** 2).mean()) * 100:.1f}")
    print(f"{'kind':18} {'busy s':>7} {'read %pk':>9} {'write %pk':>9} {'read GB':>8} {'write GB':>8}  share")
    traffic = {k: (fit["read"][col[k]] + fit["write"][col[k]]) * total[k] / 1e9 * peak for k in kinds}
    all_traffic = sum(traffic.values())
    for k in sorted(kinds, key=lambda k: -traffic[k]):
        busy = total[k] / 1e9
        r, w = fit["read"][col[k]], fit["write"][col[k]]
        print(f"{k:18} {busy:7.2f} {r * 100:9.1f} {w * 100:9.1f} {r * busy * peak:8.0f} {w * busy * peak:8.0f}"
              f"  {traffic[k] / all_traffic * 100:5.1f}%")


def nnls(A, b, iters=3000):
    """Non-negative least squares by projected gradient (no SciPy in the venv)."""
    AtA, Atb = A.T @ A, A.T @ b
    x = np.maximum(np.linalg.lstsq(A, b, rcond=None)[0], 0)
    step = 1.0 / max(np.linalg.eigvalsh(AtA).max(), 1e-12)
    for _ in range(iters):
        x = np.maximum(x - step * (AtA @ x - Atb), 0)
    return x


if __name__ == "__main__":
    main()
