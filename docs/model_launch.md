# OS-aware model launch and distillation

`config/model_launch.json` is the central launch contract for the local Ollama teacher stack. It connects four previously separate inputs:

- `config/ollama_models.json` — local model aliases and base generation settings;
- `config/distillation.json` — student and default on-policy parameters;
- `config/document_sources.json` — allowlisted official sources for normalized settings;
- `config/emulator_matrix.json` — exact emulator/source-build/image profiles.

The launch loader never downloads models or OS images. It only validates checked-in configuration and normalized ORM JSON supplied by the operator.

## All-OS settings preload

`ModelLaunchBundle.from_json(...)` builds an `OSSettingsRegistry` from normalized `setting_variant` claims. Records are grouped by `os_family`, deduplicated, and bounded by `max_evidence_tier`.

The default launch policy requires settings for every configured family:

- `android-aosp`;
- `grapheneos`;
- `calyxos`;
- `eos`.

Startup fails when any required OS family has no normalized setting records. Missing custom-OS settings are never replaced with AOSP settings.

Before a curriculum task reaches the teacher, the loader adds a bounded `os_settings_context` to its immutable evidence. The matching setting is placed first, followed by additional same-OS records up to `max_settings_context`. Qwen3-VL therefore receives the current screenshot, exact emulator scope, task evidence, and OS-specific settings context in one structured request.

## Per-OS model settings

Each OS family may override safe `OllamaModelSpec` fields without changing the approved model aliases:

- `keep_alive`;
- `num_ctx`;
- `temperature`;
- `top_p`;
- `coordinate_scale`.

The default policy uses a smaller context for the Android 13 `/e/OS` emulator and larger contexts for current AOSP and source-built OS profiles. Model names, capabilities, loopback host restrictions, and the no-download boundary remain controlled by `ollama_models.json`.

## Per-OS distillation

Each OS family has an explicit `distillation_mode`, emulator profile list, and on-policy overrides. The checked-in configuration marks all four families as `required`, so a real run must provide an environment plugin capable of materializing every listed profile.

Disabled custom-OS profiles in the general emulator matrix are intentionally selected by the launch policy. This is an explicit request to the lab runner, not permission to substitute an AOSP image. The environment plugin must fail clearly when a pinned source build or approved image artifact is unavailable.

Training is executed separately for each OS family while sharing the same student and updater. This permits different confidence thresholds, rollout batch sizes, step limits, and seeds while preserving one final checkpoint lineage.

## Run

Create the normalized ORM export containing settings for all configured OS families, start the loopback Ollama server, install the approved aliases, and run:

```bash
python scripts/train_on_policy_ollama.py \
  --launch-config config/model_launch.json \
  --documents artifacts/normalized-documents.json \
  --student-plugin project_train:create_student \
  --environment-plugin project_lab:create_environment
```

A dry run validates configuration and writes the launch plan without contacting Ollama or starting emulators:

```bash
python scripts/train_on_policy_ollama.py \
  --launch-config config/model_launch.json \
  --documents artifacts/normalized-documents.json \
  --dry-run
```

Legacy path flags remain available as explicit overrides:

```text
--ollama-config
--distillation-config
--sources
--matrix
```

## Plugin context

The environment factory may retain the original single-argument contract:

```python
def create_environment(profile):
    ...
```

It may also accept the preloaded OS settings context:

```python
def create_environment(profile, os_settings_context):
    ...
```

The second argument is read-only normalized evidence. It must not be treated as an executable script or as authorization to bypass Android permissions.

The student plugin still receives `(student_config, output_path)`. `student_config["os_launch"]` now contains the resolved per-OS model and distillation settings so a real VLM/LLM adapter can initialize token limits, batching, and checkpoint metadata consistently.

## Outputs

The training directory now contains:

- `plan.json` — resolved all-OS launch settings and counts;
- `ollama-models.json` — model preflight results by OS family;
- `metrics-by-os.json` — separate metrics for each OS family;
- `metrics.json` — rollout-weighted aggregate metrics;
- `coverage.json` — evidence and rollout coverage ledger;
- `run.json` — final manifest with per-OS policies and selected profiles.

A required OS with missing settings, unknown profile IDs, mismatched profile ownership, absent local models, or unavailable custom-OS artifacts fails closed.
