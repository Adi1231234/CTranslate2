# The whisper-bench job image: the production venv (tools/turing/aws/requirements.txt), the fork's two Linux builds
# of ctranslate2 (tools/turing/linux/build.sh; buildspec.yml unpacks them from S3 into pkgs/), the model, the
# runner as it is now and as it was before resume.py (runner_old/, build.py takes it from commit 217574e9), and the
# benchmark scripts. CUDA comes from the host driver (ECS GPU AMI) and the venv's cuBLAS; ctranslate2 links the
# CUDA runtime statically. Build context: the zip build.py makes (tools/turing, runner_old, pkgs).
FROM public.ecr.aws/docker/library/ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive NVIDIA_DRIVER_CAPABILITIES=compute,utility PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv ca-certificates curl unzip procps \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSLo /tmp/awscli.zip https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip \
 && unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install && rm -rf /tmp/aws /tmp/awscli.zip
# Nsight Systems for box/profile.sh (2025.3.2: the CUDA 13.0 generation, as the Batch hosts' driver 580).
RUN curl -fsSLo /tmp/kr.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i /tmp/kr.deb && apt-get update && apt-get install -y --no-install-recommends nsight-systems-2025.3.2 \
 && rm -rf /var/lib/apt/lists/* /tmp/kr.deb && nsys --version
WORKDIR /opt/wb
COPY tools/turing/aws/requirements.txt /tmp/requirements.txt
RUN python3.12 -m venv venv && venv/bin/pip install -q --no-cache-dir -r /tmp/requirements.txt
RUN HF_HOME=/opt/wb/hf venv/bin/python -c \
      "from faster_whisper.utils import download_model as d; print(d('ivrit-ai/whisper-large-v3-ct2'))"
COPY pkgs/ /opt/wb/
COPY runner_old/ /opt/wb/runner_old/
COPY tools/turing/ /opt/wb/src/tools/turing/
RUN mkdir -p runner && cp src/tools/turing/runner/*.py runner/ \
 && test -f pyct2-l40/ctranslate2/BUILD.txt && test -f pyct2-l40-224/ctranslate2/BUILD.txt \
 && test -f runner/resume.py && test ! -f runner_old/resume.py
ENTRYPOINT ["bash", "/opt/wb/src/tools/turing/aws/batch/job/entry.sh"]
