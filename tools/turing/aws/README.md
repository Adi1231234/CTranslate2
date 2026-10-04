# The production runner on an AWS GPU box

How fast does the crowd-v5 pipeline (whisper large-v3, the exact decoding parameters, pipe8 + fallback, this fork)
run on an AWS GPU, and how much of the GPU does it leave idle? A throwaway EC2 box, driven from the laptop.

- `launch.ps1 [-Type g6e.xlarge] [-Az us-east-1a] [-Minutes 150]`: the box. Deep Learning Base AMI (Ubuntu 24.04),
  profile `docvoice-ec2-ssm`, tagged `Project=whisper-aws-bench`, and it terminates itself after `-Minutes`.
- `ssm.ps1 -Id <instance> -Script box\<step>.sh [-ScriptArgs ...]`: one step on the box through SSM, as root.
- `box/setup.sh <fork commit> <SSM parameter with the HF token> [units list]`: the venv (`requirements.txt`, the
  store PC's production versions), the fork, the Linux builds (`../linux/build.sh`, uploaded to
  `s3://docvoice-042984981008-code/whisper-aws-bench/`), the model, `units.json` checked against the production
  hash, and the units cached. The token is on the box only while the units are fetched.
- `box/run.sh <label> <stock | package> <runner dir> <units list> <processes> [VAR=value ...]`: one configuration,
  N runner processes at once (`MPS=1`: under NVIDIA MPS); one JSON line in `/opt/wb/results.jsonl`.

On Linux CTranslate2 loads cuBLAS with `dlopen("libcublas.so.12")`, so `run.sh` puts the venv's
`nvidia/cublas/lib` on `LD_LIBRARY_PATH` before the process starts (setting it from Python is too late).

**Beside other AWS work in the account** (the asr-training Batch queues): a standalone box, never their queues;
no resource named `asr-train*` (their submit uses the newest `asr-train` job definition); another AZ than their
running box; everything tagged.
