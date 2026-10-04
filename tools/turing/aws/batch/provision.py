"""Creates (or confirms) the whisper-bench stack: ECR repository, log group, IAM roles, launch template, compute
environment (on-demand g6e, ECS GPU AMI, min 0 vCPU: costs nothing while idle) and job queue. Idempotent.
usage: python provision.py"""
import time
from botocore.exceptions import ClientError
from settings import (COMPUTE_ENV, ECR_REPO, INSTANCE_TYPES, LAUNCH_TEMPLATE, LOG_GROUP, MAX_VCPUS, QUEUE, ROOT_GB,
                      TAGS, client, tag_list)
import iam


def _exists(fn, code):
    try:
        fn()
    except ClientError as e:
        if e.response["Error"]["Code"] != code:
            raise


def registry_and_logs():
    _exists(lambda: client("ecr").create_repository(repositoryName=ECR_REPO, tags=tag_list()),
            "RepositoryAlreadyExistsException")
    _exists(lambda: client("logs").create_log_group(logGroupName=LOG_GROUP, tags=TAGS), "ResourceAlreadyExistsException")
    client("logs").put_retention_policy(logGroupName=LOG_GROUP, retentionInDays=30)


def launch_template():
    """Root disk size, encrypted, IMDSv2. Batch adds the instance type, network, profile and its ECS join."""
    data = {"BlockDeviceMappings": [{"DeviceName": "/dev/xvda", "Ebs": {
                "VolumeSize": ROOT_GB, "VolumeType": "gp3", "DeleteOnTermination": True, "Encrypted": True}}],
            "MetadataOptions": {"HttpEndpoint": "enabled", "HttpTokens": "required", "HttpPutResponseHopLimit": 2},
            "TagSpecifications": [{"ResourceType": "volume", "Tags": tag_list()},
                                  {"ResourceType": "instance", "Tags": tag_list()}]}
    _exists(lambda: client("ec2").create_launch_template(LaunchTemplateName=LAUNCH_TEMPLATE, LaunchTemplateData=data,
            TagSpecifications=[{"ResourceType": "launch-template", "Tags": tag_list()}]), "InvalidLaunchTemplateName.AlreadyExistsException")


def network():
    """Default-VPC subnets of every AZ that sells the instance types, and the default security group."""
    ec2 = client("ec2")
    vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    azs = {o["Location"] for o in ec2.describe_instance_type_offerings(LocationType="availability-zone",
           Filters=[{"Name": "instance-type", "Values": INSTANCE_TYPES}])["InstanceTypeOfferings"]}
    subnets = [s["SubnetId"] for s in ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc]},
               {"Name": "default-for-az", "Values": ["true"]}])["Subnets"] if s["AvailabilityZone"] in azs]
    sg = ec2.describe_security_groups(Filters=[{"Name": "vpc-id", "Values": [vpc]},
                                               {"Name": "group-name", "Values": ["default"]}])["SecurityGroups"][0]
    return subnets, sg["GroupId"]


def _wait(describe, key, name):
    """Batch has no waiter: poll until the resource is VALID (seconds)."""
    for _ in range(60):
        items = describe()[key]
        if items and items[0]["status"] == "VALID":
            return
        if items and items[0]["status"] == "INVALID":
            raise SystemExit(f"{name} INVALID: {items[0].get('statusReason')}")
        time.sleep(5)
    raise SystemExit(f"{name} not VALID after 5 minutes")


def compute_and_queue(profile):
    b = client("batch")
    subnets, sg = network()
    if not b.describe_compute_environments(computeEnvironments=[COMPUTE_ENV])["computeEnvironments"]:
        b.create_compute_environment(computeEnvironmentName=COMPUTE_ENV, type="MANAGED", state="ENABLED", tags=TAGS,
            computeResources={"type": "EC2", "allocationStrategy": "BEST_FIT_PROGRESSIVE", "minvCpus": 0,
                "maxvCpus": MAX_VCPUS, "instanceTypes": INSTANCE_TYPES, "subnets": subnets,
                "securityGroupIds": [sg], "instanceRole": profile, "tags": {**TAGS, "Name": COMPUTE_ENV},
                "ec2Configuration": [{"imageType": "ECS_AL2023_NVIDIA"}],
                "launchTemplate": {"launchTemplateName": LAUNCH_TEMPLATE, "version": "$Default"}})
        print("created compute environment", COMPUTE_ENV, "in", len(subnets), "subnets")
    _wait(lambda: b.describe_compute_environments(computeEnvironments=[COMPUTE_ENV]), "computeEnvironments", COMPUTE_ENV)
    if not b.describe_job_queues(jobQueues=[QUEUE])["jobQueues"]:
        b.create_job_queue(jobQueueName=QUEUE, state="ENABLED", priority=1, tags=TAGS,
                           computeEnvironmentOrder=[{"order": 1, "computeEnvironment": COMPUTE_ENV}])
        print("created job queue", QUEUE)
    _wait(lambda: b.describe_job_queues(jobQueues=[QUEUE]), "jobQueues", QUEUE)


if __name__ == "__main__":
    registry_and_logs()
    profile = iam.instance_profile()
    print("job role", iam.job_role(), "| codebuild role", iam.codebuild_role())
    launch_template()
    compute_and_queue(profile)
    print("stack ready:", COMPUTE_ENV, QUEUE)
