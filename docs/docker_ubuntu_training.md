# Ubuntu Docker Compose training

This repository includes a CUDA training image and Docker Compose services for
preparing Uni-GUI-OpenMobile data, generating Qwen screen chat data, and training
the two GUI-Owl LoRA adapters on an Ubuntu host.

## Host requirements

Use an Ubuntu machine with Docker Engine, the Docker Compose v2 plugin, an NVIDIA
driver, and NVIDIA Container Toolkit configured for Docker. The container uses
CUDA; the host does not need a separate Python, PyTorch, or torchvision install.

Check the host first:

```bash
nvidia-smi
docker --version
docker compose version
```

After NVIDIA Container Toolkit is configured, Docker must be able to expose the
GPU to containers. The Compose `train-*` and `gpu-shell` services request all
NVIDIA GPUs through a device reservation.

## Configure environment

From the repository root:

```bash
cp .env.example .env
```

Set these values as needed:

```dotenv
# Optional for public Hugging Face downloads, recommended for rate limits.
HF_TOKEN=

# Required only by the Qwen chat-generation step.
DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1

# Strongly recommended: immutable Uni-GUI-OpenMobile dataset commit.
UNIGUI_REVISION=
UNIGUI_APP=settings

SMOKE_RECORDS=8
```

Do not commit `.env`; it is excluded from the Docker build context.

## Build and verify CUDA

```bash
docker compose build

docker compose run --rm gpu-shell python - <<'PY'
import torch, torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("CUDA runtime", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0))
PY
```

The default image is `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`. The
Dockerfile keeps the image's CUDA-enabled Torch 2.5.1 and installs the matching
`torchvision==0.20.1` explicitly, avoiding the macOS-style missing torchvision
failure and avoiding an accidental CPU-only torch replacement during pip
resolution.

A different official PyTorch image can be supplied at build time:

```bash
docker compose build \
  --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
```

When changing the Torch version, update the torchvision pin in
`Dockerfile.train` to the matching release.

## Prepare Uni-GUI-OpenMobile

The preparation service does not require a GPU. It writes its output into the
host `./artifacts` directory through the repository bind mount.

```bash
docker compose run --rm prepare-unigui
```

Expected output:

```text
artifacts/training/gui-policy-unigui-openmobile.jsonl
artifacts/training/gui-policy-unigui-openmobile.jsonl.manifest.json
```

For reproducible training, set `UNIGUI_REVISION` to an immutable dataset commit
before this step.

## Generate Qwen chat data

With `DASHSCOPE_API_KEY` configured:

```bash
docker compose run --rm generate-chat
```

This produces:

```text
artifacts/training/chat-screen-qa-qwen.jsonl
```

The Qwen API key is passed as an environment variable and is not written to the
training records.

## Smoke-test GUI-policy training

Run a small CUDA training pass before launching the complete adapter job:

```bash
docker compose run --rm train-gui-smoke
```

Change the sample count in `.env` with `SMOKE_RECORDS`, for example:

```dotenv
SMOKE_RECORDS=16
```

The smoke adapter is written to:

```text
artifacts/adapters/gui_policy-smoke
```

## Train the adapters

GUI policy:

```bash
docker compose run --rm train-gui
```

Chat:

```bash
docker compose run --rm train-chat
```

Outputs remain on the Ubuntu host:

```text
artifacts/adapters/gui_policy
artifacts/adapters/chat
```

Hugging Face and Torch caches use named Docker volumes (`hf-cache` and
`torch-cache`) so rebuilding/rerunning containers does not redownload every
model artifact.

## Interactive debugging

CPU/no-GPU shell:

```bash
docker compose run --rm shell
```

CUDA shell:

```bash
docker compose run --rm gpu-shell
```

Inside the CUDA shell, useful checks are:

```bash
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
wc -l artifacts/training/gui-policy-unigui-openmobile.jsonl
```

The LoRA trainer still accepts `--device cpu`, but training a 2B vision-language
model on CPU is intended only for debugging and is generally impractical for a
full run.
