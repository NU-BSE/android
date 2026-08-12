# GGUF export and Mac transfer for compact GUI-Owl students

This stage runs after knowledge distillation. It converts each complete
Hugging Face Qwen3-VL student into the two files required by llama.cpp-style
multimodal runtimes and then packages the source weights plus mobile GGUF files
for transfer off a disposable Vast.ai host.

The three expected source checkpoints are:

```text
artifacts/students/gui-owl-0.5b-distilled/
artifacts/students/gui-owl-1b-distilled/
artifacts/students/gui-owl-1.5b-distilled/
```

## Why there are two GGUF files

Qwen3-VL deployment uses a language-model GGUF and a separate multimodal
projector/vision GGUF (`mmproj`). llama.cpp's Hugging Face converter exposes
`--mmproj` for supported vision architectures. The language model can then be
quantized independently with `llama-quantize`.

For mobile transfer the default artifact pair is:

```text
gui-owl-<size>-q4_0.gguf
mmproj-gui-owl-<size>-bf16.gguf
```

The BF16 language GGUF is kept on the training host as a reproducible
quantization source but is not copied to the Mac by default because the full HF
checkpoint already preserves the source weights.

## 1. Export on Vast.ai

Pull the latest distillation branch first:

```bash
cd /root/android
git checkout feat/gui-owl-student-distillation
git pull
```

Export one student:

```bash
bash scripts/export_gui_owl_gguf.sh 1b
```

Or export all three after all distillation jobs are complete:

```bash
bash scripts/export_gui_owl_gguf.sh all
```

The script uses `~/llama.cpp` by default. If that checkout does not exist it
clones the official `ggml-org/llama.cpp` repository and checks out the pinned
`b9637` release. Override this when validating a newer llama.cpp fix:

```bash
LLAMA_CPP_REF=<tag-or-commit> \
  bash scripts/export_gui_owl_gguf.sh 1b
```

If a pre-existing llama.cpp checkout should be used:

```bash
LLAMA_CPP_DIR=/opt/llama.cpp \
  bash scripts/export_gui_owl_gguf.sh 1b
```

The script deliberately does not alter Python dependencies unless requested.
The Vast PyTorch image already contains Torch/CUDA. If llama.cpp's converter
reports missing Python packages, either install its requirements explicitly or
allow the script to do so:

```bash
LLAMA_CPP_INSTALL_DEPS=1 \
  bash scripts/export_gui_owl_gguf.sh 1b
```

`llama-quantize` is built CPU-only because GGUF quantization does not need the
training GPU. `cmake` is required. If it is absent on the selected Vast image,
install the host package before exporting.

### Quantization choices

The mobile default is Q4_0:

```bash
GGUF_QUANTS=Q4_0 bash scripts/export_gui_owl_gguf.sh all
```

Generate both Q4_0 and Q6_K for quality/size benchmarking:

```bash
GGUF_QUANTS='Q4_0 Q6_K' bash scripts/export_gui_owl_gguf.sh all
```

The projector remains BF16 by default. Each student receives an export manifest
with the exact llama.cpp commit, byte sizes, and SHA-256 hashes.

Expected outputs include:

```text
artifacts/gguf/gui-owl-0.5b-bf16.gguf
artifacts/gguf/gui-owl-0.5b-q4_0.gguf
artifacts/gguf/mmproj-gui-owl-0.5b-bf16.gguf
artifacts/gguf/gui-owl-0.5b-export-manifest.json

artifacts/gguf/gui-owl-1b-bf16.gguf
artifacts/gguf/gui-owl-1b-q4_0.gguf
artifacts/gguf/mmproj-gui-owl-1b-bf16.gguf
artifacts/gguf/gui-owl-1b-export-manifest.json

artifacts/gguf/gui-owl-1.5b-bf16.gguf
artifacts/gguf/gui-owl-1.5b-q4_0.gguf
artifacts/gguf/mmproj-gui-owl-1.5b-bf16.gguf
artifacts/gguf/gui-owl-1.5b-export-manifest.json
```

Do not delete the Hugging Face checkpoints after GGUF conversion. They are the
source-of-truth weights for future re-quantization or converter changes.

## 2. Build one transfer bundle per model size

On Vast.ai:

```bash
bash scripts/package_gui_owl_students.sh all
```

This creates:

```text
artifacts/transfer/gui-owl-0.5b-mac-bundle.tar
artifacts/transfer/gui-owl-0.5b-mac-bundle.tar.sha256
artifacts/transfer/gui-owl-1b-mac-bundle.tar
artifacts/transfer/gui-owl-1b-mac-bundle.tar.sha256
artifacts/transfer/gui-owl-1.5b-mac-bundle.tar
artifacts/transfer/gui-owl-1.5b-mac-bundle.tar.sha256
```

Each default bundle contains:

```text
full distilled Hugging Face checkpoint
Q4_0 language GGUF
BF16 mmproj GGUF
GGUF export manifest
student architecture config
```

It intentionally excludes the intermediate BF16 language GGUF. Include it only
when there is a concrete need to preserve that exact converter output:

```bash
INCLUDE_FULL_GGUF=1 \
  bash scripts/package_gui_owl_students.sh all
```

For a deployment-only bundle that omits the Hugging Face source weights:

```bash
INCLUDE_HF_CHECKPOINT=0 \
  bash scripts/package_gui_owl_students.sh all
```

For this project the default should remain `INCLUDE_HF_CHECKPOINT=1` until the
models have been fully evaluated on-device.

## 3. Pull the weights to the Mac

The Mac needs the Vast public SSH host/IP and the external SSH port shown by the
Vast connection dialog. In the local repository checkout:

```bash
git checkout feat/gui-owl-student-distillation
git pull
```

Then run:

```bash
bash scripts/pull_gui_owl_students_to_mac.sh \
  --host <VAST_HOST> \
  --port <VAST_SSH_PORT>
```

The default remote repository is `/root/android` and the default Mac destination
is `~/Models/gui-owl`. Override either when needed:

```bash
bash scripts/pull_gui_owl_students_to_mac.sh \
  --host <VAST_HOST> \
  --port <VAST_SSH_PORT> \
  --remote-root /workspace/android \
  --dest "$HOME/Models/creepy-gui-owl"
```

Pull only one size:

```bash
bash scripts/pull_gui_owl_students_to_mac.sh \
  --host <VAST_HOST> \
  --port <VAST_SSH_PORT> \
  --student 1b
```

The Mac script uses resumable `rsync --partial` when rsync is installed on both
ends; otherwise it falls back to `scp`. Every bundle is verified with SHA-256
before extraction. A failed hash stops the script.

Verified files are extracted under:

```text
~/Models/gui-owl/extracted/0.5b/
~/Models/gui-owl/extracted/1b/
~/Models/gui-owl/extracted/1.5b/
```

Keep the downloaded `.tar` bundles until local inference tests pass.

## 4. Destruction gate for the Vast instance

Do not destroy the Vast instance until all of the following are true:

1. all requested student distillation manifests exist;
2. all requested GGUF export manifests exist;
3. the Mac pull script reports a successful SHA-256 check for every bundle;
4. the extracted Mac directories contain both the HF checkpoint and the
   Q4_0/mmproj GGUF pair;
5. at least one local llama.cpp/llama.rn smoke inference has loaded each pair.

Stopping the instance before the hashes are verified risks losing the only copy
of the newly trained weights.
