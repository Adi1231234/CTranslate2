#!/bin/bash
# On the benchmark box (as root, through ssm.ps1): uv and the production venv (../requirements.txt), the fork at a
# commit, the two Linux builds from S3, the model, Adi's HF token from Parameter Store only until the units are
# cached, units.json (the runner's own list_units, checked against the production file's hash), the units cached.
# usage: setup.sh <fork commit> <SSM parameter holding the HF token> [units list, default scale/units_real.txt]
set -euo pipefail
B=/opt/wb; S3=s3://docvoice-042984981008-code/whisper-aws-bench; COMMIT=$1; PARAM=$2
mkdir -p $B && cd $B
command -v uv >/dev/null || [ -x /root/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH=/root/.local/bin:$PATH
[ -d src ] || git clone -q --filter=blob:none -b store-pc https://github.com/Adi1231234/CTranslate2 src
git -C src fetch -q origin store-pc && git -C src checkout -q "$COMMIT"
LIST=${3:-src/tools/turing/scale/units_real.txt}
[ -x venv/bin/python ] || uv venv -q -p 3.12 venv
uv pip install -q -p venv/bin/python -r src/tools/turing/aws/requirements.txt
for p in pyct2-l40 pyct2-l40-224; do [ -d $p ] || aws s3 cp --only-show-errors $S3/$p.tgz - | tar xz; done
rm -rf runner runner_old && mkdir -p runner runner_old
cp src/tools/turing/runner/*.py runner/                              # with resume.py (store-pc 2f74fea and later)
git -C src archive 217574e9 tools/turing/runner | tar x --strip-components=3 -C runner_old   # before it
HF_HOME=$B/hf venv/bin/python -c "from faster_whisper.utils import download_model as d; d('ivrit-ai/whisper-large-v3-ct2')"
aws ssm get-parameter --with-decryption --name "$PARAM" --query Parameter.Value --output text > runner/hf_token.txt
chmod 600 runner/hf_token.txt
trap 'rm -f $B/runner/hf_token.txt' EXIT
venv/bin/python src/tools/turing/aws/box/units_json.py runner B3ECAFAE
cp runner/units.json runner_old/
HF_HOME=$B/hf venv/bin/python src/tools/turing/scale/prefetch.py runner "$LIST" cache 8
echo "cache: $(ls cache | wc -l) units, $(du -sh cache | cut -f1)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nproc; free -g | head -2
