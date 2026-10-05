"""Uploads the runner of a commit as s3://$BUCKET/$S3_PREFIX/runner-<commit>.tgz (top folder runner-<commit>), which
job/entry.sh fetches when an experiment line names that runner folder: a runner change needs no image build.
Prints the folder name.
usage: python pack_runner.py [commit, default HEAD]"""
import io, os, subprocess, sys, tarfile
from settings import BUCKET, S3_PREFIX, client

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
rev = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
git = lambda *a: subprocess.run(["git", "-C", SRC, *a], capture_output=True, check=True).stdout
name = "runner-" + git("rev-parse", "--short=8", rev).decode().strip()
out = io.BytesIO()
with tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", rev, "tools/turing/runner"))) as src, \
        tarfile.open(fileobj=out, mode="w:gz") as dst:
    for m in src.getmembers():
        if m.isfile():
            data = src.extractfile(m).read()
            m.name = name + m.name[len("tools/turing/runner"):]
            dst.addfile(m, io.BytesIO(data))
client("s3").put_object(Bucket=BUCKET, Key=f"{S3_PREFIX}/{name}.tgz", Body=out.getvalue())
print(name)
