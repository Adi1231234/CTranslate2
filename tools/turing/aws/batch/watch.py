"""Follows one whisper-bench job: its status changes and new log lines (CloudWatch), until it ends or --minutes
pass; then the results from S3. Batch has no completion event for the laptop, so this polls every 20 s.
usage: python watch.py <job id> [--minutes 8]"""
import argparse, json, time
from settings import BUCKET, LOG_GROUP, S3_PREFIX, client

p = argparse.ArgumentParser(); p.add_argument("job"); p.add_argument("--minutes", type=float, default=8)
a = p.parse_args()
batch, logs = client("batch"), client("logs")
status, token, end = None, None, time.time() + a.minutes * 60
while True:
    job = batch.describe_jobs(jobs=[a.job])["jobs"][0]
    if job["status"] != status:
        status = job["status"]
        print(time.strftime("%H:%M:%S"), status, job.get("statusReason", ""), flush=True)
    stream = job.get("container", {}).get("logStreamName")
    if stream:
        kw = {"nextToken": token} if token else {"startFromHead": True}
        try:
            r = logs.get_log_events(logGroupName=LOG_GROUP, logStreamName=stream, **kw)
            for e in r["events"]:
                print("  |", e["message"][:300], flush=True)
            token = r["nextForwardToken"]
        except logs.exceptions.ResourceNotFoundException:
            pass
    if status in ("SUCCEEDED", "FAILED") or time.time() > end:
        break
    time.sleep(20)
if status in ("SUCCEEDED", "FAILED"):
    reason = job.get("container", {}).get("reason") or job.get("statusReason")
    print("ended", status, reason or "", "exit", job.get("container", {}).get("exitCode"))
    try:
        body = client("s3").get_object(Bucket=BUCKET, Key=f"{S3_PREFIX}/results/{a.job}/results.jsonl")["Body"].read()
        for line in body.decode().splitlines():
            print(json.dumps(json.loads(line)))
    except client("s3").exceptions.NoSuchKey:
        print("no results.jsonl")
