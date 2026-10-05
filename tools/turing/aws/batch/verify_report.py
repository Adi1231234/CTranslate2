"""The bit-for-bit verification's verdicts (experiments/verify1.txt and verify2.txt): fetches both jobs' rows and
check outputs (fetch_results.py), then prints every comparison (../../scale/compare.py): stock against stock (the
run-to-run control), the production package in stock's mode against stock (--strict: the sampled rows too), the
production configuration against stock, stock as shipped against stock with the full context (what the length rule
changes), the production configuration and stock against the production reference (compare_ref.py's job); then the
logits, seeded-draw and kernel bit-check lines of the script steps.
usage: python verify_report.py <verify1 job id> <verify2 job id> <dest dir>"""
import os, re, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
REFERENCE_JOB = re.search(r'REFERENCE_JOB = "([^"]+)"', open(os.path.join(here, "compare_ref.py")).read()).group(1)
compare = os.path.join(here, "..", "..", "scale", "compare.py")
job1, job2, dest = sys.argv[1:4]
jobs = {job1: ["stockA", "stockB", "forkS", "prod", "stock224"], job2: ["stockN", "forkN", "prodN"]}
for job, labels in {**jobs, REFERENCE_JOB: ["fullctx"]}.items():
    where = os.path.join(dest, job[:8])
    if not os.path.isdir(os.path.join(where, "out", labels[0])):
        subprocess.run([sys.executable, "fetch_results.py", job, where, "--out", *labels], cwd=here, check=True,
                       stdout=subprocess.DEVNULL)
out = lambda job, label: os.path.join(dest, job[:8], "out", label)
pairs = [("stock run-to-run (stockA, stockB)", out(job1, "stockA"), out(job1, "stockB"), True),
         ("fork in stock's mode (stockA, forkS)", out(job1, "stockA"), out(job1, "forkS"), True),
         ("production configuration (stockA, prod)", out(job1, "stockA"), out(job1, "prod"), False),
         ("length rule only (stock224, stockA)", out(job1, "stock224"), out(job1, "stockA"), False),
         ("stock against the reference (fullctx, stockA)", out(REFERENCE_JOB, "fullctx"), out(job1, "stockA"), False),
         ("production against the reference (fullctx, prod)", out(REFERENCE_JOB, "fullctx"), out(job1, "prod"), False),
         ("60 new units, fork in stock's mode (stockN, forkN)", out(job2, "stockN"), out(job2, "forkN"), True),
         ("60 new units, production configuration (stockN, prodN)", out(job2, "stockN"), out(job2, "prodN"), False)]
for name, ref, new, strict in pairs:
    result = subprocess.run([sys.executable, compare, ref, new] + (["--strict"] if strict else []),
                            capture_output=True, text=True, encoding="utf-8")
    print(f"== {name}{' --strict' if strict else ''}\n{result.stdout.strip()}{result.stderr.strip()}", flush=True)
for job, script, pattern in ((job1, "scriptv", r"IDENTICAL|DIFFERENT|clips|calls|: \d+ s"),
                             (job2, "scriptp", r"^=== |TOTAL|mismatch|differ|identical|exact|probes from")):
    path = os.path.join(dest, job[:8], script + ".txt")
    lines = open(path, encoding="utf-8").read().splitlines() if os.path.exists(path) else ["(no output)"]
    print(f"== {script}\n" + "\n".join(l for l in lines if re.search(pattern, l, re.IGNORECASE)))
