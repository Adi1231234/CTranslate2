"""Creates (or confirms) the whisper-bench stack: ECR repository, log group, IAM roles, launch template, and per
fleet (settings.FLEETS) a compute environment (on-demand, ECS GPU AMI, min 0 vCPU: free while idle) and its job queue.
Idempotent.
usage: python provision.py"""
import time
from botocore.exceptions import ClientError
from settings import (ECR_REPO, FLEET_ALSO, FLEETS, LAUNCH_TEMPLATE, LOG_GROUP, MAX_VCPUS, MAX_VCPUS_ALSO, ROOT_GB,
                      TAGS, client, fleet_name, tag_list)
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


def network(instance_type):
    """Default-VPC subnets of every AZ that sells the instance type, and the default security group."""
    ec2 = client("ec2")
    vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    azs = {o["Location"] for o in ec2.describe_instance_type_offerings(LocationType="availability-zone",
           Filters=[{"Name": "instance-type", "Values": [instance_type]}])["InstanceTypeOfferings"]}
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


def compute_and_queue(profile, fleet):
    """The fleet's compute environment and its job queue, both named fleet_name(fleet)."""
    b, name, itype = client("batch"), fleet_name(fleet), FLEETS[fleet][0]
    subnets, sg = network(itype)
    types, max_vcpus = [itype, *FLEET_ALSO.get(fleet, [])], MAX_VCPUS_ALSO if fleet in FLEET_ALSO else MAX_VCPUS
    if not b.describe_compute_environments(computeEnvironments=[name])["computeEnvironments"]:
        b.create_compute_environment(computeEnvironmentName=name, type="MANAGED", state="ENABLED", tags=TAGS,
            computeResources={"type": "EC2", "allocationStrategy": "BEST_FIT_PROGRESSIVE", "minvCpus": 0,
                "maxvCpus": max_vcpus, "instanceTypes": types, "subnets": subnets,
                "securityGroupIds": [sg], "instanceRole": profile, "tags": {**TAGS, "Name": name},
                "ec2Configuration": [{"imageType": "ECS_AL2023_NVIDIA"}],
                "launchTemplate": {"launchTemplateName": LAUNCH_TEMPLATE, "version": "$Default"}})
        print("created compute environment", name, itype, "in", len(subnets), "subnets")
    _wait(lambda: b.describe_compute_environments(computeEnvironments=[name]), "computeEnvironments", name)
    if not b.describe_job_queues(jobQueues=[name])["jobQueues"]:
        b.create_job_queue(jobQueueName=name, state="ENABLED", priority=1, tags=TAGS,
                           computeEnvironmentOrder=[{"order": 1, "computeEnvironment": name}])
        print("created job queue", name)
    _wait(lambda: b.describe_job_queues(jobQueues=[name]), "jobQueues", name)


if __name__ == "__main__":
    registry_and_logs()
    profile = iam.instance_profile()
    print("job role", iam.job_role(), "| codebuild role", iam.codebuild_role())
    launch_template()
    for fleet in FLEETS:
        compute_and_queue(profile, fleet)
    print("stack ready:", ", ".join(fleet_name(f) for f in FLEETS))
