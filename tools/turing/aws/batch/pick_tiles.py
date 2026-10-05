"""Reads decoder_gemm_probe's output (a job's scriptg.txt) and picks, per kind of decoder product, the tile with the
least total time over the row counts a stream step has (M >= 80), against cuBLAS; prints CT2_DECODER_TILES. A tile
with a mismatched value is never picked.
usage: python pick_tiles.py <scriptg.txt>"""
import re, sys

kinds = {(3840, 1280): "qkv", (1280, 1280): "o", (5120, 1280): "ffn1", (51872, 1280): "vocab"}
text = open(sys.argv[1], encoding="utf-8").read()
bad = set()
for line in text.splitlines():
    if "mismatched values by tile" in line:
        for name, count in re.findall(r"(\d+x\d+/\d+) (\d+)", line):
            if int(count):
                bad.add(name)
picked, kind, tiles, rows = [], None, [], []


def pick():
    if not kind or not rows:
        return
    totals = [sum(r[i] for r in rows) for i in range(len(rows[0]))]          # cuBLAS first
    best = min((t for t in range(1, len(totals)) if tiles[t - 1] not in bad), key=lambda t: totals[t], default=None)
    gain = totals[0] / totals[best] if best else 0
    print(f"{kind}: cuBLAS {totals[0]:.0f} us, best {tiles[best - 1] if best else '-'} {totals[best] if best else 0:.0f} us"
          f" ({gain:.2f}x) over M {[m for m in ms]}")
    if best and gain > 1.05:
        picked.append(f"{kind}={tiles[best - 1]}")


ms = []
for line in text.splitlines():
    head = re.match(r"\s*(\d+) x (\d+) \([\d.]+ MB\): us at M = cuBLAS \| tiles (.*)", line)
    groups = re.match(r"\s*1280 x 5120 .*groups of 40 rows", line)
    if head or groups:
        pick()
        rows, ms = [], []
        kind = kinds.get((int(head.group(1)), int(head.group(2)))) if head else "ffn2"
        tiles = head.group(3).split() if head else tiles
        continue
    row = re.match(r"\s*(?:M\s+(\d+)|(\d) x 40):\s+([\d.]+) \|((?:\s+[\d.]+)+)", line)
    if row and kind:
        m = int(row.group(1)) if row.group(1) else 40 * int(row.group(2))
        if m >= 80:
            ms.append(m)
            rows.append([float(row.group(3))] + [float(x) for x in row.group(4).split()])
pick()
print("CT2_DECODER_TILES=" + ",".join(picked))
