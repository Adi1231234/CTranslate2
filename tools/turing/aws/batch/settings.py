"""Names and sizes of the whisper-bench AWS Batch stack (README.md). Every resource is named whisper-bench* and
tagged Project=whisper-aws-bench, apart from the asr-training stack in the same account (never touch asr-train*)."""
import os
import boto3

# The Batch stack's region: us-east-1, or another one (WB_REGION) when its GPUs are sold out; the image (ECR), the
# bucket and the image build stay in HOME_REGION, which every stack reads from.
HOME_REGION = "us-east-1"
REGION = os.environ.get("WB_REGION", HOME_REGION)
NAME = "whisper-bench"
BUCKET = "docvoice-042984981008-code"
S3_PREFIX = "whisper-aws-bench"                 # packages, data/ (units.json, cache/), build/, results/
TAGS = {"Project": "whisper-aws-bench"}
ROLE_INSTANCE, ROLE_JOB, ROLE_CODEBUILD = f"{NAME}-instance", f"{NAME}-job", f"{NAME}-codebuild"
JOB_DEF, LAUNCH_TEMPLATE = NAME, NAME
ECR_REPO, CODEBUILD, LOG_GROUP = NAME, f"{NAME}-image", f"/{NAME}"
# One compute environment and one job queue (same name) per instance type, so a job always lands on the type it
# asked for and every measurement is on a known machine. Job size: (vCPU, memory MiB) as ECS can place it.
# g7e: the RTX PRO 6000 Blackwell Server (sm_120, 96 GB GDDR7), the store PC's GPU generation.
FLEETS = {"g6e": ("g6e.xlarge", 4, 28 * 1024), "g6e2x": ("g6e.2xlarge", 8, 56 * 1024),
          "g7e": ("g7e.2xlarge", 8, 56 * 1024)}
# Larger sizes of the same single GPU a fleet may also take when its own type is sold out (more vCPU, dearer).
# Set 5.10 on the g7e fleets and the eu-north-1 / eu-central-1 g6e fleets (the us-east g6e ones stay g6e.xlarge:
# the asr-train queues wait for that capacity there).
FLEET_ALSO = {"g7e": ["g7e.4xlarge"], "g6e": ["g6e.2xlarge"]}
MAX_VCPUS = 8                                   # per fleet: two g6e.xlarge or one g6e.2xlarge at most
MAX_VCPUS_ALSO = 16                             # a fleet with larger sizes: one of them at most
ROOT_GB = 100                                   # the image (~6 GB) and its layers, the cache, the outputs


def fleet_name(fleet):
    return f"{NAME}-{fleet}"                     # the compute environment and its job queue


def client(service, region=None):
    home = service in ("s3", "ecr", "codebuild")
    return boto3.client(service, region_name=region or (HOME_REGION if home else REGION))


def account():
    return client("sts").get_caller_identity()["Account"]


def tag_list(extra=None):
    return [{"Key": k, "Value": v} for k, v in {**TAGS, "Name": NAME, **(extra or {})}.items()]
