"""Compare the PTX of two ctranslate2.dll builds kernel by kernel (cuobjdump -ptx), by .entry name: does a
build reproduce the official wheel's kernels? cub/thrust put the build's arch list into their namespace
(CUB_200700_530_..._860_NS, which also changes the mangled length prefix) and block labels are numbered by
the function's index in its module, so both are normalized.
usage: ptx_compare.py <cuobjdump.exe> <official dll> <built dll> [--show <entry substring>]"""
import re, subprocess, sys, hashlib, difflib

ENTRY = r"(?:\.visible\s+)?\.entry\s+(\S+?)\s*\((.*?)(?=\n\s*(?:\.visible\s+)?\.entry\s|\n\s*\.func\s|\nFatbin |\Z)"


def entries(dump, dll):
    ptx = subprocess.run([dump, "-ptx", dll], capture_output=True, text=True, errors="replace").stdout
    ptx = re.sub(r"\d+(CUB|THRUST)_\d+(?:_\d{3})+_NS", r"\1_NS", ptx)
    ptx = re.sub(r"(\$L__BB|__local_depot)\d+", r"\1", ptx)
    out = {}
    for m in re.finditer(ENTRY, ptx, re.S):
        out.setdefault(m.group(1), set()).add(m.group(2).rstrip().removesuffix("//").rstrip())
    return out


args = sys.argv[1:]
show = args[args.index("--show") + 1] if "--show" in args else None
dump, a_dll, b_dll = args[:3]
a, b = entries(dump, a_dll), entries(dump, b_dll)
common = a.keys() & b.keys()
h = lambda bodies: {hashlib.sha256(x.encode()).hexdigest() for x in bodies}
diff = sorted(n for n in common if h(a[n]) != h(b[n]))
print(f"official entries {len(a)}, built {len(b)}, common {len(common)}, identical {len(common) - len(diff)}, "
      f"differ {len(diff)}; only in built {len(b.keys() - a.keys())}, only in official {len(a.keys() - b.keys())}")
for n in diff:
    print("DIFF", n[:160])
for n in sorted(a.keys() - b.keys()):
    print("ONLY-OFFICIAL", n[:160])
if show:
    n = next(n for n in diff if show in n)
    x, y = sorted(a[n])[0].splitlines(), sorted(b[n])[0].splitlines()
    print(f"--- {n[:120]}: official {len(x)} lines, built {len(y)} lines")
    for line in list(difflib.unified_diff(x, y, lineterm="", n=1))[:60]:
        print(line[:160])
