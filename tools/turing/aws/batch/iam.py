"""The stack's IAM roles. The job runs in HOST network mode, so the container can also use the instance role: that
role holds no secrets. The job needs no HF token at all (units.json and the units come from S3)."""
import json, time
from botocore.exceptions import ClientError
from settings import BUCKET, S3_PREFIX, ECR_REPO, LOG_GROUP, REGION, ROLE_INSTANCE, ROLE_JOB, ROLE_CODEBUILD, \
    account, client, tag_list

MANAGED = "arn:aws:iam::aws:policy/"


def _trust(service):
    return json.dumps({"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}]})


def _role(name, service, managed=(), inline=None):
    iam = client("iam")
    try:
        iam.get_role(RoleName=name)
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise
        iam.create_role(RoleName=name, AssumeRolePolicyDocument=_trust(service), Tags=tag_list())
        print("created role", name)
    for arn in managed:
        iam.attach_role_policy(RoleName=name, PolicyArn=MANAGED + arn)
    if inline:
        iam.put_role_policy(RoleName=name, PolicyName=f"{name}-inline", PolicyDocument=json.dumps(inline))
    return f"arn:aws:iam::{account()}:role/{name}"


def _s3(read=True, write_results=False):
    arn = f"arn:aws:s3:::{BUCKET}"
    st = [{"Effect": "Allow", "Action": "s3:ListBucket", "Resource": arn,
           "Condition": {"StringLike": {"s3:prefix": [f"{S3_PREFIX}/*"]}}}]
    if read:
        st.append({"Effect": "Allow", "Action": "s3:GetObject", "Resource": f"{arn}/{S3_PREFIX}/*"})
    if write_results:
        st.append({"Effect": "Allow", "Action": "s3:PutObject", "Resource": f"{arn}/{S3_PREFIX}/results/*"})
    return st


def instance_profile():
    """ECS agent + image pull + awslogs (AmazonEC2ContainerServiceforEC2Role); nothing else."""
    _role(ROLE_INSTANCE, "ec2.amazonaws.com", ["service-role/AmazonEC2ContainerServiceforEC2Role"])
    iam = client("iam")
    try:
        iam.create_instance_profile(InstanceProfileName=ROLE_INSTANCE, Tags=tag_list())
        iam.add_role_to_instance_profile(InstanceProfileName=ROLE_INSTANCE, RoleName=ROLE_INSTANCE)
        print("created instance profile", ROLE_INSTANCE)
        time.sleep(15)                                   # IAM propagation before Batch validates it
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
    return f"arn:aws:iam::{account()}:instance-profile/{ROLE_INSTANCE}"


def job_role():
    return _role(ROLE_JOB, "ecs-tasks.amazonaws.com",
                 inline={"Version": "2012-10-17", "Statement": _s3(read=True, write_results=True)})


def codebuild_role():
    repo = f"arn:aws:ecr:{REGION}:{account()}:repository/{ECR_REPO}"
    logs = f"arn:aws:logs:{REGION}:{account()}:log-group:{LOG_GROUP}*"
    return _role(ROLE_CODEBUILD, "codebuild.amazonaws.com", inline={"Version": "2012-10-17", "Statement": _s3() + [
        {"Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
        {"Effect": "Allow", "Resource": repo, "Action": [
            "ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
            "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"]},
        {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"], "Resource": logs}]})
