# Ubuntu Docker Compose training

This repository includes a CUDA image and Docker Compose services for Uni-GUI data preparation, Qwen screen-chat generation, the two GUI-Owl LoRA adapters, and the calibrated compact-student pipeline.

## Host requirements

Use Ubuntu with Docker Engine, Docker Compose v2, an NVIDIA driver, and NVIDIA Container Toolkit configured for Docker. Verify the host with:

```bash
nvidia-smi
docker --version
docker compose version
```

GPU services reserve **one** NVIDIA GPU. The current trainers are single-device, so reserving every visible GPU would waste capacity on a multi-GPU host.

## Environment

```bash
cp .env.example .env
```

Relevant values are:

```dotenv
HF_TOKEN=
DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
UNIGUI_REVISION=
UNIGUI_APP=settings
SMOKE_RECORDS=8
RECOVERY_RECORDS=50000
RECOVERY_MIN_RECORDS=10000
STUDENT_SIZE=1.5b
CALIBRATION_RECORDS=64
DISTILL_MAX_RECORDS=
DISTILL_MAX_UPDATES=
DISTILL_INTERMEDIATE_UPDATES=750
DISTILL_RESUME=0
```

Do not commit `.env`.

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

The default image is `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`. It retains the CUDA Torch supplied by that image, pins `torchvision==0.20.1`, and installs Hugging Face `datasets` for recovery-corpus preparation.

## Prepare specialized data

```bash
docker compose run --rm prepare-unigui
docker compose run --rm generate-chat
```

Outputs:

```text
artifacts/training/gui-policy-unigui-openmobile.jsonl
artifacts/training/chat-screen-qa-qwen.jsonl
```

`generate-chat` requires `DASHSCOPE_API_KEY`; the API key remains an environment variable and is not written to the generated records.

## Train the 2B teacher adapters

Run a bounded GUI-policy smoke job first:

```bash
docker compose run --rm train-gui-smoke
```

Then train both adapters:

```bash
docker compose run --rm train-gui
docker compose run --rm train-chat
```

Outputs:

```text
artifacts/adapters/gui_policy
artifacts/adapters/chat
```

## Prepare broad recovery data

The v2 compact-student pipeline does not attempt structural recovery from only the Android Settings specialization corpus. Prepare the configured broad instruction corpus first:

```bash
docker compose run --rm prepare-recovery
```

Outputs:

```text
artifacts/training/gui-owl-recovery.jsonl
artifacts/training/gui-owl-recovery.jsonl.manifest.json
```

The progressive runner refuses to begin recovery when the required corpus is absent.

## Calibrate pruning

```bash
docker compose run --rm calibrate-student
```

This writes:

```text
artifacts/calibration/gui-owl-compression.json
```

The progressive runner normalizes the calibration before building students so grouped-query attention head groups and the three Qwen3-VL deepstack vision taps stay aligned during 0.5B pruning.

## Progressive compact students

Establish 1.5B first:

```bash
STUDENT_SIZE=1.5b docker compose run --rm progressive-student
```

Only after that checkpoint passes its generation gate, run:

```bash
STUDENT_SIZE=1b docker compose run --rm progressive-student
STUDENT_SIZE=0.5b docker compose run --rm progressive-student
```

Compression order:

```text
2B/28 -> 24 -> recover -> 20 -> recover -> 16 -> recover ~= 1.5B
1.5B/16 -> 12 -> recover -> 9 -> recover -> 6 -> recover ~= 1B
1.5B/16 -> structured text/vision width prune -> recover ~= 0.5B
```

Each recovered student then receives student-sampled K3 reverse-KL and final low-learning-rate GUI/chat specialization. Failed free-generation gates stop the cascade.

A plumbing-only smoke run is available with:

```bash
STUDENT_SIZE=1.5b \
DISTILL_MAX_RECORDS=16 \
DISTILL_MAX_UPDATES=2 \
  docker compose run --rm progressive-student-smoke
```

A two-update smoke run is not a quality run; it is intended to expose loading, tensor-shape, processor, and CUDA problems cheaply.

Accepted v2 outputs are written separately from the old collapsed checkpoints:

```text
artifacts/students/v2/gui-owl-1.5b-distilled
artifacts/students/v2/gui-owl-1b-distilled
artifacts/students/v2/gui-owl-0.5b-distilled
```

Do not GGUF-export a checkpoint unless `distillation-manifest.json` contains `"accepted": true`; the export script enforces that by default.

## Caches and debugging

Hugging Face and Torch caches use the `hf-cache` and `torch-cache` named volumes. Repository artifacts remain under host `./artifacts` through the bind mount.

```bash
docker compose run --rm shell
docker compose run --rm gpu-shell
```

Useful checks inside the GPU shell:

```bash
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
wc -l artifacts/training/gui-policy-unigui-openmobile.jsonl
wc -l artifacts/training/gui-owl-recovery.jsonl
```

The LoRA trainer can still run with `--device cpu` for debugging, but full VLM training on CPU is impractical.
