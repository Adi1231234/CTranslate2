"""The same experiments queued on several fleets or regions when capacity is short: once one job starts, the others
are cancelled (or terminated, if they started too), so only one is paid for. Batch sends the laptop no event, so
this polls every 30 s. A copy that ends without ever starting (cancelled while it waited) leaves the race; the
others go on. Prints the job that runs.
usage: python race.py <region>/<job id> <region>/<job id> ..."""
import sys, time
from settings import client

jobs = [arg.split("/", 1) for arg in sys.argv[1:]]
clients = {region: client("batch", region) for region, _ in jobs}
while jobs:
    found = {(r, j): clients[r].describe_jobs(jobs=[j])["jobs"][0] for r, j in jobs}
    status = {k: d["status"] for k, d in found.items()}
    # Started: it got a container (startedAt), whatever its status now.
    started = [k for k, d in found.items() if d.get("startedAt") or d["status"] in ("STARTING", "RUNNING")]
    if started:
        winner = started[0]
        for region, job in jobs:
            if (region, job) == winner:
                continue
            if status[(region, job)] in ("STARTING", "RUNNING"):
                clients[region].terminate_job(jobId=job, reason=f"race lost to {winner[1]}")
            elif status[(region, job)] not in ("SUCCEEDED", "FAILED"):
                clients[region].cancel_job(jobId=job, reason=f"race lost to {winner[1]}")
        print(time.strftime("%H:%M:%S"), "running:", "/".join(winner), {"/".join(k): v for k, v in status.items()},
              flush=True)
        break
    gone = [k for k, s in status.items() if s in ("SUCCEEDED", "FAILED")]
    if gone:
        print(time.strftime("%H:%M:%S"), "left the race without starting:", ["/".join(k) for k in gone], flush=True)
        jobs = [[r, j] for r, j in jobs if (r, j) not in gone]
        continue
    time.sleep(30)
else:
    print(time.strftime("%H:%M:%S"), "no copy started", flush=True)
