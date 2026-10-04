"""Names and sizes of the whisper-bench AWS Batch stack (README.md). Every resource is named whisper-bench* and
tagged Project=whisper-aws-bench, apart from the asr-training stack in the same account (never touch asr-train*)."""
import boto3

REGION = "us-east-1"
NAME = "whisper-bench"
BUCKET = "docvoice-042984981008-code"
S3_PREFIX = "whisper-aws-bench"                 # packages, data/ (units.json, cache/), build/, results/
TAGS = {"Project": "whisper-aws-bench"}
ROLE_INSTANCE, ROLE_JOB, ROLE_CODEBUILD = f"{NAME}-instance", f"{NAME}-job", f"{NAME}-codebuild"
COMPUTE_ENV, QUEUE, JOB_DEF, LAUNCH_TEMPLATE = f"{NAME}-g6e", f"{NAME}-queue", NAME, NAME
ECR_REPO, CODEBUILD, LOG_GROUP = NAME, f"{NAME}-image", f"/{NAME}"
INSTANCE_TYPES = ["g6e.xlarge"]                 # one type: every measurement on the same machine
MAX_VCPUS = 8                                   # two g6e.xlarge at most
ROOT_GB = 100                                   # the image (~6 GB) and its layers, the cache, the outputs
JOB_VCPUS, JOB_MEMORY_MIB = 4, 28 * 1024        # g6e.xlarge: 4 vCPU, 32 GiB (ECS keeps some for itself)


def client(service):
    return boto3.client(service, region_name=REGION)


def account():
    return client("sts").get_caller_identity()["Account"]


def tag_list(extra=None):
    return [{"Key": k, "Value": v} for k, v in {**TAGS, "Name": NAME, **(extra or {})}.items()]
