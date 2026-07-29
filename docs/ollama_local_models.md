# Local Qwen3-VL and UI-TARS through Ollama

This integration runs both teacher models through the loopback Ollama API at
`http://127.0.0.1:11434`. The application rejects remote hosts, alternate
ports, URL-based screenshots and proxy forwarding.

## Roles

- `android-qwen3-vl-teacher:latest` is based on `qwen3-vl:30b`. It receives the
  screenshot, normalized documentation evidence, emulator scope, the current
  student proposal and the verified environment outcome. Ollama structured
  output constrains the response to the closed teacher schema.
- `android-ui-tars-critic:latest` is based on `avil/UI-TARS:latest`. It receives
  the same screenshot and independently proposes one native UI-TARS action.
  Its action is parsed as a critique only; it is never executed.

Qwen3-VL supplies the correction target. UI-TARS contributes only an agreement
score. A strong disagreement blocks the sample from updating student weights.
The Android environment receipt and immutable ORM evidence remain the sources
of truth.

## Install Ollama models

Start Ollama on its default local endpoint:

```bash
bash scripts/start_ollama_local.sh
```

In another terminal, install the model aliases:

```bash
bash scripts/setup_ollama_models.sh
```

The script performs these explicit operations:

```bash
ollama pull qwen3-vl:30b
ollama pull avil/UI-TARS:latest
ollama create android-qwen3-vl-teacher -f ollama/Modelfile.qwen3-vl-teacher
ollama create android-ui-tars-critic -f ollama/Modelfile.ui-tars-critic
```

The repository never pulls models during import, tests or training startup.
Model installation is a separate operator action because the artifacts are
large and their licenses/provenance must be reviewed.

## Preflight

```bash
python scripts/check_ollama_models.py \
  --config config/ollama_models.json
```

For a real multimodal request to each model:

```bash
python scripts/check_ollama_models.py \
  --config config/ollama_models.json \
  --smoke
```

Preflight calls Ollama `/api/tags` and `/api/show` and requires both models to
report `completion` and `vision`. A community UI-TARS package that reports only
text capability is rejected rather than silently used as a text-only critic.
ByteDance's UI-TARS repository warns that quantized GGUF performance is not
guaranteed, so emulator benchmark gates remain mandatory.

## Coordinator adapters

```python
from android_drlqn import JsonGroundingVLM, OllamaRuntimeBundle

bundle = OllamaRuntimeBundle.from_json("config/ollama_models.json")
bundle.preflight()

chat_model = bundle.chat_model
vlm = JsonGroundingVLM(bundle.multimodal_generator)
teacher = bundle.teacher
```

`chat_model` implements `LocalChatModel`. `multimodal_generator` implements the
existing `MultimodalGenerator` protocol, and `teacher` implements
`TeacherEvaluator`.

## On-policy distillation

```bash
python scripts/train_on_policy_ollama.py \
  --documents artifacts/normalized-documents.json \
  --student-plugin project_train:create_student \
  --environment-plugin project_lab:create_environment
```

The emulator plugin must return timestamp-aligned screenshots using either:

- a path under one of `server.allowed_image_roots`;
- a `file://` URI under an allowed root;
- a `data:image/...;base64,...` URI;
- a `base64:...` value.

HTTP image URLs are rejected. Screenshot files are size-limited before they are
sent to the loopback model server.

## Configuration

`config/ollama_models.json` controls model aliases, context windows, keep-alive,
image roots and ensemble weighting. The endpoint is deliberately fixed to a
loopback host and port 11434 by validation code; changing JSON to a remote host
or another port fails before any request is sent.

## Operational notes

- Ollama's REST vision API accepts base64 image data in the message `images`
  field.
- Qwen3-VL uses `think: false`; hidden reasoning is not collected or stored.
- Structured output is used only for Qwen3-VL. UI-TARS is parsed using its
  native `Action_Summary` / `Action` grammar.
- Model responses never receive Android execution tools.
- Model residency is controlled through `keep_alive`; operators must still
  enforce GPU memory, thermal and concurrency limits appropriate to their host.

Primary references:

- https://docs.ollama.com/api/introduction
- https://docs.ollama.com/capabilities/vision
- https://docs.ollama.com/capabilities/structured-outputs
- https://docs.ollama.com/api-reference/show-model-details
- https://ollama.com/library/qwen3-vl
- https://github.com/bytedance/UI-TARS
