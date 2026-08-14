# GGUF export and Mac transfer for accepted GUI-Owl v2 students

GGUF conversion is now strictly downstream of the free-generation gate. The v1
Q4_0 files were hash-correct while their source students were collapsed, so
conversion integrity is no longer treated as a model-quality signal.

The v2 source checkpoints are resolved from `config/gui_owl_students.json` and
default to:

```text
artifacts/students/v2/gui-owl-0.5b-distilled/
artifacts/students/v2/gui-owl-1b-distilled/
artifacts/students/v2/gui-owl-1.5b-distilled/
```

Each source must contain `distillation-manifest.json` with:

```json
{"accepted": true}
```

Otherwise `scripts/export_gui_owl_gguf.sh` exits before conversion. A debug-only
`GGUF_ALLOW_UNACCEPTED=1` override exists for diagnosing converters; never use
that output as a deployment candidate.

## Why there are two GGUF files

Qwen3-VL deployment uses a language-model GGUF and a separate multimodal
projector/vision GGUF (`mmproj`). The language model is quantized independently;
the projector remains BF16 by default.

The mobile pair is:

```text
gui-owl-<size>-q4_0.gguf
mmproj-gui-owl-<size>-bf16.gguf
```

## 1. Export on the GPU host

After the v2 distillation gate passes:

```bash
bash scripts/export_gui_owl_gguf.sh 1.5b
bash scripts/export_gui_owl_gguf.sh 1b
bash scripts/export_gui_owl_gguf.sh 0.5b
```

Or:

```bash
bash scripts/export_gui_owl_gguf.sh all
```

The script uses `~/llama.cpp` by default. If absent it clones llama.cpp and
checks out the configured `LLAMA_CPP_REF` (default `b9637`). Conversion
requirements are not installed unless `LLAMA_CPP_INSTALL_DEPS=1` is explicitly
set, so the training Torch/CUDA environment is not silently replaced.

Generate Q4_0 plus Q6_K for comparison with:

```bash
GGUF_QUANTS='Q4_0 Q6_K' bash scripts/export_gui_owl_gguf.sh all
```

Default outputs now live under:

```text
artifacts/gguf/v2/
```

Each export manifest records the accepted HF source, exact llama.cpp commit,
conversion precision, file sizes and SHA-256 hashes.

## 2. Package transfer bundles

```bash
bash scripts/package_gui_owl_students.sh all
```

Default bundles are written to:

```text
artifacts/transfer/v2/gui-owl-0.5b-mac-bundle.tar
artifacts/transfer/v2/gui-owl-1b-mac-bundle.tar
artifacts/transfer/v2/gui-owl-1.5b-mac-bundle.tar
```

Each has a sibling `.tar.sha256`. By default the bundle contains the accepted
Hugging Face checkpoint, Q4_0 language GGUF, BF16 mmproj, export manifest and
student config. Set `INCLUDE_HF_CHECKPOINT=0` only for a deployment-only copy;
set `INCLUDE_FULL_GGUF=1` only when the unquantized language GGUF is needed.

## 3. Pull to macOS

From a local checkout of this branch:

```bash
bash scripts/pull_gui_owl_students_to_mac.sh \
  --host <VAST_HOST> \
  --port <VAST_SSH_PORT>
```

Defaults:

```text
remote repo:     /root/android
remote bundles:  artifacts/transfer/v2
local root:      ~/Models/gui-owl-v2
```

Use `--student 1b` (or another size) to download only one model. The script uses
resumable `rsync --partial` when available on both ends and otherwise falls back
to `scp`. Every tar is verified with SHA-256 before extraction.

## 4. Instance destruction gate

Do not destroy the GPU instance until all requested models satisfy all of these:

1. the v2 distillation manifest says `accepted: true`;
2. the GGUF export manifest exists and all file hashes match;
3. the Mac pull script verifies the bundle SHA-256;
4. both Q4_0 and mmproj files are present after extraction;
5. local llama.cpp/llama.rn free-generation smoke inference is coherent.

The last condition matters: hash verification proves that the transferred bytes
match the export, not that the student learned a usable policy.
