"""Registers the whisper-bench job definition for an image tag and submits one job with an experiment list
(job/entry.sh). The job has a hard time limit (Batch kills it after --minutes) and no retries, so a stuck
benchmark cannot keep a GPU billing. Prints the job id.
--privileged runs the container privileged: the GPU's hardware counters (box/gpumetrics.sh) need it, as the
driver keeps them to administrators (ERR_NVGPUCTRPERM).
usage: python submit.py <image tag> <experiments file> [--fleet g6e|g6e2x] [--name n] [--minutes 90] [--privileged]"""
import argparse
from settings import (BUCKET, ECR_REPO, FLEETS, JOB_DEF, LOG_GROUP, REGION, ROLE_JOB, S3_PREFIX, TAGS, account,
                      client, fleet_name)

p = argparse.ArgumentParser()
p.add_argument("tag"); p.add_argument("experiments"); p.add_argument("--name", default="bench")
p.add_argument("--minutes", type=int, default=90); p.add_argument("--fleet", default="g6e", choices=FLEETS)
p.add_argument("--privileged", action="store_true")
a = p.parse_args()
_, vcpus, memory = FLEETS[a.fleet]
lines = [l.strip() for l in open(a.experiments, encoding="utf-8") if l.strip() and not l.startswith("#")]
batch = client("batch")
jd = batch.register_job_definition(jobDefinitionName=f"{JOB_DEF}-{a.fleet}", type="container", tags=TAGS, propagateTags=True,
    retryStrategy={"attempts": 1}, timeout={"attemptDurationSeconds": a.minutes * 60},
    containerProperties={
        "image": f"{account()}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO}:{a.tag}",
        "jobRoleArn": f"arn:aws:iam::{account()}:role/{ROLE_JOB}",
        "resourceRequirements": [{"type": "VCPU", "value": str(vcpus)},
                                 {"type": "MEMORY", "value": str(memory)}, {"type": "GPU", "value": "1"}],
        "linuxParameters": {"sharedMemorySize": 4096, "initProcessEnabled": True}, "privileged": a.privileged,
        "environment": [{"name": "BUCKET", "value": BUCKET}, {"name": "S3_PREFIX", "value": S3_PREFIX},
                        {"name": "AWS_DEFAULT_REGION", "value": REGION}],
        "logConfiguration": {"logDriver": "awslogs", "options": {
            "awslogs-group": LOG_GROUP, "awslogs-region": REGION, "awslogs-stream-prefix": "job"}}})
job = batch.submit_job(jobName=f"{JOB_DEF}-{a.name}", jobQueue=fleet_name(a.fleet), jobDefinition=jd["jobDefinitionArn"],
                       containerOverrides={"environment": [{"name": "EXPERIMENTS", "value": "\n".join(lines)}]})
print(job["jobId"], f"{jd['jobDefinitionName']}:{jd['revision']}", a.fleet, f"{len(lines)} experiments, limit {a.minutes} min")
