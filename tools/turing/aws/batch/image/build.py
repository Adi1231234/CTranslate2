"""Builds the whisper-bench image in CodeBuild and pushes it to ECR (costs cents). The context is the COMMITTED
tools/turing of this checkout plus the runner of commit 217574e9 as runner_old/; the tag comes from that content
and the two packages' S3 ETags, so an unchanged input never builds twice. CodeBuild has no completion event for
the laptop: this polls the build every 20 s. Prints the tag.
usage: python build.py [fork checkout, default: the checkout this file is in]"""
import hashlib, io, os, subprocess, sys, tarfile, time, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from botocore.exceptions import ClientError
from settings import BUCKET, CODEBUILD, ECR_REPO, LOG_GROUP, REGION, ROLE_CODEBUILD, S3_PREFIX, account, client, TAGS

OLD_RUNNER = "217574e9"                       # the runner before resume.py, for the stock-identity runs
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
git = lambda *a: subprocess.run(["git", "-C", SRC, *a], capture_output=True, check=True).stdout


def context_zip():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rev, path, dest in (("HEAD", "tools/turing", "tools/turing"),
                                (OLD_RUNNER, "tools/turing/runner", "runner_old")):
            with tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", rev, path))) as t:
                for m in t.getmembers():
                    if m.isfile():
                        z.writestr(dest + m.name[len(path):], t.extractfile(m).read())
    return out.getvalue()


def tag():
    s3 = client("s3")
    etags = [s3.head_object(Bucket=BUCKET, Key=f"{S3_PREFIX}/{p}.tgz")["ETag"] for p in ("pyct2-l40", "pyct2-l40-224")]
    key = git("rev-parse", "HEAD:tools/turing") + OLD_RUNNER.encode() + "".join(etags).encode()
    return "img-" + hashlib.sha256(key).hexdigest()[:12]


def project(ecr_uri):
    spec = {"name": CODEBUILD, "source": {"type": "S3", "location": f"{BUCKET}/{S3_PREFIX}/build/none.zip",
                                          "buildspec": "tools/turing/aws/batch/image/buildspec.yml"},
            "artifacts": {"type": "NO_ARTIFACTS"},
            "environment": {"type": "LINUX_CONTAINER", "image": "aws/codebuild/standard:7.0",
                            "computeType": "BUILD_GENERAL1_MEDIUM", "privilegedMode": True,
                            "environmentVariables": [{"name": "ECR_URI", "value": ecr_uri},
                                                     {"name": "BUCKET", "value": BUCKET},
                                                     {"name": "S3_PREFIX", "value": S3_PREFIX}]},
            "serviceRole": f"arn:aws:iam::{account()}:role/{ROLE_CODEBUILD}", "timeoutInMinutes": 60,
            "logsConfig": {"cloudWatchLogs": {"status": "ENABLED", "groupName": LOG_GROUP, "streamName": "codebuild"}}}
    cb = client("codebuild")
    if cb.batch_get_projects(names=[CODEBUILD])["projects"]:
        cb.update_project(**spec)
    else:
        cb.create_project(**spec, tags=[{"key": k, "value": v} for k, v in TAGS.items()])


def main():
    t, ecr_uri = tag(), f"{account()}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO}"
    try:
        client("ecr").describe_images(repositoryName=ECR_REPO, imageIds=[{"imageTag": t}])
        print("image exists:", t); return t
    except ClientError as e:
        if e.response["Error"]["Code"] != "ImageNotFoundException":
            raise
    key = f"{S3_PREFIX}/build/{t}.zip"
    client("s3").put_object(Bucket=BUCKET, Key=key, Body=context_zip())
    project(ecr_uri)
    build = client("codebuild").start_build(projectName=CODEBUILD, sourceLocationOverride=f"{BUCKET}/{key}",
        environmentVariablesOverride=[{"name": "IMAGE_TAG", "value": t, "type": "PLAINTEXT"}])["build"]
    print("build", build["id"], "tag", t, flush=True)
    phase = None
    while True:
        b = client("codebuild").batch_get_builds(ids=[build["id"]])["builds"][0]
        if b["currentPhase"] != phase:
            phase = b["currentPhase"]; print(time.strftime("%H:%M:%S"), phase, flush=True)
        if b["buildComplete"]:
            print("status", b["buildStatus"]); sys.exit(0 if b["buildStatus"] == "SUCCEEDED" else 1)
        time.sleep(20)


if __name__ == "__main__":
    print(main())
