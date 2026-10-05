"""A job's rows against the production reference: fetch_results.py with the rows of the given configurations, then
../../scale/compare.py of each against the reference rows, the fullctx package's run on the L40S (job d654b84b,
4.10.2026: pyct2-l40, 1 process, pipe8, the fallback inline; every "IDENTICAL" in ../README.md is against it),
fetched once into <dest>/../fullctx. Sampled rows (a temperature above 0) are counted apart: no two unseeded runs
draw the same; a seeded run's sampled rows are compared with another seeded run's (an experiments `compare` line).
A RUN_FALLBACK=skip run lists the 2 fallback clips decoded at T = 0 as differing ("fallback_skipped").
usage: python compare_ref.py <job id> <dest dir> <label> [label ...]"""
import os, subprocess, sys

REFERENCE_JOB = "d654b84b-c0f0-49f2-8d76-cbbed9bba03d"
here = os.path.dirname(os.path.abspath(__file__))
job, dest, labels = sys.argv[1], sys.argv[2], sys.argv[3:]
ref_dest = os.path.join(os.path.dirname(os.path.abspath(dest)), "fullctx")
if not os.path.isdir(os.path.join(ref_dest, "out", "fullctx")):
    subprocess.run([sys.executable, "fetch_results.py", REFERENCE_JOB, ref_dest, "--out", "fullctx"], cwd=here, check=True)
subprocess.run([sys.executable, "fetch_results.py", job, dest, "--out", *labels], cwd=here, check=True)
compare = os.path.join(here, "..", "..", "scale", "compare.py")
for label in labels:
    out = subprocess.run([sys.executable, compare, os.path.join(ref_dest, "out", "fullctx"),
                          os.path.join(dest, "out", label)], capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    lines = (out.stdout + out.stderr).strip().splitlines()
    print(f"== {label}: " + " | ".join(lines[-3:]))
