"""The same experiments queued on several fleets or regions when capacity is short: once one job starts, the others
are cancelled (or terminated, if they started too), so only one is paid for. Batch sends the laptop no event, so
this polls every 30 s. Prints the job that runs.
usage: python race.py <region>/<job id> <region>/<job id> ..."""
import sys, time
from settings import client

jobs = [arg.split("/", 1) for arg in sys.argv[1:]]
clients = {region: client("batch", region) for region, _ in jobs}
while True:
    status = {(r, j): clients[r].describe_jobs(jobs=[j])["jobs"][0]["status"] for r, j in jobs}
    started = [k for k in status if status[k] in ("STARTING", "RUNNING", "SUCCEEDED", "FAILED")]
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
    time.sleep(30)
