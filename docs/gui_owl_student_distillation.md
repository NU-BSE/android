# GUI-Owl compact-student distillation v2

This pipeline compresses the trained GUI-Owl 2B teacher into full ~1.5B, ~1B,
and ~0.5B Qwen3-VL checkpoints. It replaces the original one-shot recipe after
live llama.cpp tests showed that the Q4_0 files were intact while their source
students had collapsed during training.

Observed v1 failure:

```text
size   language params   Q4_0    supervised CE   mean teacher KL   greedy output
0.5B   357M              236MB   5.68            8.13              token soup
1.0B   613M              411MB   5.03            6.62              token soup
1.5B   1.12B             681MB   2.36            2.03              immediate EOS
```

The 1.5B result is the key regression: teacher-forced CE looked much better, but
free-running generation emitted `<|im_end|>` immediately. V2 therefore does not
accept CE or GGUF integrity as proof that a student is usable.

## Inputs

Specialized behavior still comes from:

```text
mPLUG/GUI-Owl-1.5-2B-Instruct
├── gui_policy adapter <- Uni-GUI-OpenMobile trajectories
└── chat adapter       <- Qwen3.7-Plus screen-chat data
```

Structural recovery additionally uses a broad text-instruction corpus, mixed
with the existing multimodal specialized records. The default preparer streams
a filtered subset of `HuggingFaceH4/ultrachat_200k`, pins dataset revision
`8049631c405ae6576f93f445c6b8166f76f5505a`, validates alternating user/assistant
turns, requires an assistant-final dialogue, bounds conversation size, removes
exact duplicates, and writes:

```text
artifacts/training/gui-owl-recovery.jsonl
artifacts/training/gui-owl-recovery.jsonl.manifest.json
```

Prepare the default 50k records with:

```bash
python -m pip install -e '.[data]'
python scripts/prepare_gui_owl_recovery.py
```

Or with Docker Compose:

```bash
docker compose run --rm prepare-recovery
```

Each output line has `messages`, `teacher_mode: "base"`, and source provenance.
The broad corpus is text-only; 1.5B recovery explicitly opts into the existing
GUI/image and screen-chat corpora while retaining the broad corpus at weight
2.0. The trainer rejects a 100% text-only structural recovery mix even when the
vision tower is frozen, so the trainable text stack cannot silently be recovered
outside its deployment regime. The progressive runner also refuses to start
paid recovery if the required broad corpus is missing.

## Architecture targets

```text
1.5B: text 2048 / FFN 6144 / 20 blocks, original 24-block vision tower
1.0B: text 2048 / FFN 6144 /  6 blocks, original 24-block vision tower
0.5B: text 1024 / FFN 3072 / 16 blocks, 12-block 768-wide vision tower
```

The 20-layer 1.5B checkpoint has 1,724,844,032 parameters, a 14.99% relative
error against the nominal 1.5B target, so it remains inside the configured 15%
parameter tolerance.

The language GGUF can report fewer parameters because llama.cpp exports the
multimodal projector separately; the Hugging Face build manifest remains the
source of truth for the complete student count.

## 1. Calibrate pruning

Run the trained 2B teacher over a bounded GUI/chat calibration sample:

```bash
python scripts/calibrate_gui_owl_compression.py \
  --device cuda \
  --max-records-per-corpus 64
```

This writes:

```text
artifacts/calibration/gui-owl-compression.json
```

The manifest records activation/residual saliency for text layers, residual
channels, MLP neurons, query/KV heads, vision layers, vision channels, vision
heads, and vision MLP neurons. The 0.5B build is allowed only when calibration
actually exercised the vision tower.

Before a progressive build, the runner normalizes text head selections at the
**GQA group** level. Qwen3-VL uses two query heads per KV head in both the source
and 0.5B target. Query and KV heads are therefore never independently top-k
pruned: a retained KV head keeps its corresponding query-head pair.

The old `first N dimensions` prefix-copy rule is removed. Every width-changing
tensor must have a structured mapping. Unknown shape changes fail rather than
silently leaving random weights.

Recommended text-layer maps from a calibration manifest are accepted only when
they preserve the live boundary/tail-protection contract. A stale manifest that
drops the protected final text blocks is ignored for that depth and the builder
recomputes the selection from the manifest's recorded saliency scores.

## 2. Progressive depth compression

Depth-only students are not cut to final size in one step:

```text
2B / 28 layers
  -> 24 -> recovery
  -> 20 -> recovery        ~= 1.5B

accepted 1.5B / 20
  -> 12 -> recovery
  ->  9 -> recovery
  ->  6 -> recovery        ~= 1B
```

Layer retention comes from the calibration saliency manifest while original
teacher-layer lineage is preserved across intermediate checkpoints. Each next
reduction starts from the preceding recovered model.

The 0.5B target remains 16 text layers, so it does **not** initialize from the
six-layer 1B checkpoint. It starts from the accepted 20-layer 1.5B checkpoint;
the 0.5B build performs its final 20 -> 16 text-layer selection together with
structured pruning of text residual width, MLP neurons, GQA head groups, vision
depth/width/heads/MLPs, and patch-merger channels.

## 3. Recovery stage

Every structural change is followed by update-based recovery rather than one
`epochs=1` pass. Defaults:

```text
supervised CE       0.25
forward teacher KL  1.00
hidden-state KD     0.15
learning rate       3e-6
final-stage budget  4000 optimizer updates
gradient accumulate 8
```

Input embeddings and the tied LM head are frozen. The unchanged vision tower is
also frozen for same-width 1B/1.5B recovery. For the 0.5B hidden-width change,
hidden-state KD uses a temporary learned projection into teacher width.

The final structural recovery is not resume-compatible with a legacy
`accepted: true` manifest that measured zero GUI records. Under `--resume`, the
runner requires `final_evaluation.gui_records > 0` before reusing that checkpoint,
so an old text-only depth-20 recovery cannot silently skip the repaired stage.

Manifests report optimizer updates, dataset passes, completion tokens and
student-sampled tokens. `records_per_epoch` plus `epochs=1` is no longer the
training-duration contract.

## 4. On-policy autoregressive KD

After teacher-forced recovery, the student samples its own responses. The
teacher evaluates those same student-generated prefixes and K3 reverse-KL is
applied per sampled token:

```text
delta = log p_teacher(y_t | y_<t) - log p_student(y_t | y_<t)
K3    = exp(delta) - delta - 1
```

A smaller supervised/forward-KL anchor remains. Default student-sampled response
length is bounded to 0..64 new tokens so first-token EOS remains observable.

## 5. Specialization

Only after recovery and on-policy KD does the final low-LR stage focus on the
Uni-GUI GUI-policy and generated screen-chat corpora. Android specialization is
therefore downstream of basic generative recovery.

## 6. Generation gates

Initialization, periodic checkpoints and final checkpoints are tested with
free-running greedy generation. The manifest records:

```text
first_token_eos_rate
empty_generation_rate
median_generated_tokens
mean_supervised_ce
mean_teacher_kl
gui_records
gui_parse_rate
```

Default structural acceptance:

```text
first-token EOS <= 5%
empty output    <= 5%
median output   >= 8 tokens
```

Final specialization additionally requires at least 50% valid GUI-Owl action
parsing on its held-out GUI sample. A failed gate retains the final debug model
and the strongest periodic checkpoint as a sibling `*-best` directory, then
exits non-zero so the cascade stops.

## Full GPU training wrappers: Hugging Face weights only

For a rented GPU host, use the model-specific wrappers below. They perform all
required preflight checks, reuse or prepare the recovery corpus, reuse or run
calibration, execute the full progressive KD path, verify the final generation
gate, and package **only the accepted Hugging Face checkpoint**. They never call
llama.cpp and never create GGUF files.

Run 1.5B first:

```bash
bash scripts/train_gui_owl_1_5b_full.sh
```

Then 1B and 0.5B, both of which require the accepted 1.5B checkpoint:

```bash
bash scripts/train_gui_owl_1b_full.sh
bash scripts/train_gui_owl_0_5b_full.sh
```

Each wrapper defaults to `RESUME=1`, so accepted progressive checkpoints are
reused after an interruption. Useful overrides are:

```bash
DEVICE=cuda \
RESUME=1 \
PACKAGE_WEIGHTS=1 \
RECOVERY_MAX_RECORDS=50000 \
CALIBRATION_RECORDS=64 \
DISTILL_INTERMEDIATE_UPDATES=750 \
  bash scripts/train_gui_owl_1_5b_full.sh
```

`DISTILL_MAX_RECORDS` and `DISTILL_MAX_UPDATES` may be set for bounded debugging,
but leaving them empty selects the configured full update budgets.

Accepted final checkpoints are:

```text
artifacts/students/v2/gui-owl-1.5b-distilled/
artifacts/students/v2/gui-owl-1b-distilled/
artifacts/students/v2/gui-owl-0.5b-distilled/
```

With `PACKAGE_WEIGHTS=1`, an uncompressed transfer tar is created under:

```text
artifacts/transfer/v2-hf/
```

For example:

```text
artifacts/transfer/v2-hf/gui-owl-1.5b-hf-v2.tar
artifacts/transfer/v2-hf/gui-owl-1.5b-hf-v2.tar.sha256
artifacts/transfer/v2-hf/gui-owl-1.5b-hf-v2-files.sha256
```

The helper can also package an already-accepted checkpoint without retraining:

```bash
bash scripts/package_gui_owl_hf_weights.sh 1.5b
bash scripts/package_gui_owl_hf_weights.sh 1b
bash scripts/package_gui_owl_hf_weights.sh 0.5b
```

The tar is intentionally uncompressed: safetensors are already dense binary
weights, so gzip usually spends rental-host CPU time for little transfer-size
benefit. The tar checksum and per-file checksums are written before transfer.

## Recommended run order without wrappers

```bash
# CPU/network: prepare broad recovery data
python scripts/prepare_gui_owl_recovery.py

# GPU: calibrate the trained 2B teacher
python scripts/calibrate_gui_owl_compression.py --device cuda

# GPU: always establish 1.5B first
python scripts/run_gui_owl_progressive_distillation.py --student 1.5b --device cuda

# These require the accepted 1.5B predecessor
python scripts/run_gui_owl_progressive_distillation.py --student 1b   --device cuda
python scripts/run_gui_owl_progressive_distillation.py --student 0.5b --device cuda
```

The progressive runner automatically makes GQA head selections group-consistent
before the first build. `--resume` reuses accepted checkpoints, but the final
structural recovery additionally needs measured GUI records so a legacy
text-only recovery is not treated as reusable.

For a plumbing-only smoke run:

```bash
python scripts/run_gui_owl_progressive_distillation.py \
  --student 1.5b \
  --device cuda \
  --max-records-per-corpus 16 \
  --max-updates 2 \
  --intermediate-recovery-updates 2
```

A two-update smoke run is not a quality run and will normally fail its generation
gate. Use it only to expose loading, tensor-shape, processor and CUDA problems.

## Docker Compose

On an Ubuntu Docker host:

```bash
docker compose run --rm prepare-recovery
docker compose run --rm calibrate-student

STUDENT_SIZE=1.5b docker compose run --rm progressive-student
STUDENT_SIZE=1b   docker compose run --rm progressive-student
STUDENT_SIZE=0.5b docker compose run --rm progressive-student
```

The Compose GPU reservation requests one GPU, not every GPU on the host. On
Vast.ai, where the provider already starts the workload inside the provider
container, run the Python or full-wrapper scripts directly instead of
Docker-in-Docker.

## Outputs and GGUF boundary

V2 outputs are isolated from the collapsed v1 checkpoints:

```text
artifacts/students/v2/gui-owl-1.5b-distilled/
artifacts/students/v2/gui-owl-1b-distilled/
artifacts/students/v2/gui-owl-0.5b-distilled/
```

Do not run GGUF conversion unless the final directory contains:

```json
{"accepted": true}
```

The full training wrappers stop before this conversion boundary. The existing
`scripts/export_gui_owl_gguf.sh` remains an explicit later/off-GPU step for an
accepted checkpoint.

## Validation boundary

CI checks Python syntax, dependency-light pruning/GQA/generation-gate contracts,
all full-wrapper/package shell syntax, other shell syntax, and
`docker compose config -q`. It does not download the 2B model, download the
recovery dataset, execute calibration hooks on live weights, run CUDA KD,
validate every 0.5B tensor mapping against a live model, or perform llama.cpp
conversion. Run a bounded GPU calibration/build before committing to a full
paid compression job.
