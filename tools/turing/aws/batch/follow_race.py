"""Waits until race.py names the copy of a job that started (in its output file), then follows that job with
watch.py. Nothing signals the file's change to a Windows shell here, so this reads it every 10 s.
usage: python follow_race.py <race.py's output file> [--minutes 150]"""
import os, re, subprocess, sys, time

race = sys.argv[1]
minutes = sys.argv[sys.argv.index("--minutes") + 1] if "--minutes" in sys.argv else "150"
here = os.path.dirname(os.path.abspath(__file__))
while True:
    text = open(race).read() if os.path.exists(race) else ""
    m = re.search(r"running: ([\w-]+)/([\w-]+)", text)
    if m:
        break
    if "no copy started" in text:
        sys.exit("no copy of the job started")
    time.sleep(10)
region, job = m.groups()
print(f"race won by {region}/{job}", flush=True)
subprocess.run([sys.executable, "watch.py", job, "--minutes", minutes], cwd=here,
               env=dict(os.environ, WB_REGION=region, PYTHONUNBUFFERED="1"))
