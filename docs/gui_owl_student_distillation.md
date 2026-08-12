# GUI-Owl compact-student distillation v2

This pipeline compresses the trained GUI-Owl 2B teacher into full ~1.5B, ~1B,
and ~0.5B Qwen3-VL checkpoints. It replaces the original one-shot distillation
recipe after live llama.cpp tests showed that all three exported Q4_0 artifacts
were intact but the underlying students had collapsed during training.

Observed failure from the v1 run:

```text
size   language params   Q4_0    supervised CE   mean teacher KL   greedy output
0.5B   357M              236MB   5.68            8.13              token soup
1.0B   613M              411MB   5.03            6.62              token soup
1.5B   1.12B             681MB   2.36            2.03              immediate EOS
```

The 1.5B result is the important regression: teacher-forced CE was much better,
but free-running generation emitted `<|im_end|>` immediately. Therefore v2 does
not accept CE or GGUF integrity as proof that a student is usable.

## Inputs

The specialized teacher behavior still comes from:

```text
mPLUG/GUI-Owl-1.5-2B-Instruct
├── gui_policy adapter <- Uni-GUI-OpenMobile trajectories
└── chat adapter       <- Qwen3.7-Plus screen-chat data
```

The v2 recovery stage additionally requires a broad recovery corpus at:

```text
artifacts/training/gui-owl-recovery.jsonl
```

Each line uses the same `{"messages": [...]}` chat-completion shape. An optional
`teacher_mode` may be `base`, `gui_policy`, or `chat`. The configured broad
corpus defaults to `base`, meaning teacher adapters are disabled for that
record. This corpus should contain diverse language/instruction and multimodal
examples; the small Settings specialization corpus is not accepted as the only
source for structural recovery.

The progressive runner fails before paid training when this required broad
corpus is absent.

## Architecture targets

```text
1.5B: text 2048 / FFN 6144 / 16 blocks, original 24-block vision tower
1.0B: text 2048 / FFN 6144 /  6 blocks, original 24-block vision tower
0.5B: text 1024 / FFN 3072 / 16 blocks, 12-block 768-wide vision tower
```

The language GGUF can report fewer parameters because llama.cpp exports the
multimodal projector separately; the Hugging Face build manifest remains the
source of truth for full student parameter count.

## 1. Calibrate pruning before changing the architecture

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
channels, MLP neurons, Q/KV heads, vision layers, vision channels, vision heads,
and vision MLP neurons. The 0.5B build is allowed only when calibration actually
exercised the vision tower.

The old `first N dimensions` prefix-copy rule is no longer used. A width-changing
build must provide structured calibration selections for every shape-changing
Qwen3-VL tensor. An unrecognized mapping fails instead of silently leaving a
random tensor in the student.

## 2. Progressive depth compression

Depth-only students are not cut from 28 layers to their final depth in one
operation.

```text
2B / 28 layers
  -> 24 -> recovery
  -> 20 -> recovery
  -> 16 -> recovery        ~= 1.5B

accepted 1.5B / 16
  -> 12 -> recovery
  ->  9 -> recovery
  ->  6 -> recovery        ~= 1B
```

At every step, layer retention is chosen from the calibration saliency manifest
while preserving original teacher-layer lineage. The next reduction is made
from the previous recovered checkpoint, so retained blocks see a gradually
changing residual distribution rather than an immediate 28->6 jump.

The 0.5B model keeps a 16-layer text topology and therefore does **not** use the
6-layer 1B checkpoint as its structural initialization source. It is
activation-pruned from the accepted 16-layer 1.5B checkpoint: text width/MLP/
heads and vision depth/width/MLP/heads are reduced together with consistent
indices.

## 3. Recovery stage

Each structural change is followed by update-based recovery rather than one
`epochs=1` pass. Default recovery uses:

```text
supervised CE      0.25
forward teacher KL 1.00
hidden-state KD    0.15
learning rate      3e-6
final-stage budget 4000 optimizer updates
accumulation       8
```

Input embeddings and the tied LM head are frozen. The unchanged vision tower is
frozen for the same-width 1B/1.5B students. This protects the vocabulary/output
geometry while transformer blocks recover from structural pruning.

For the 0.5B hidden-width change, hidden-state KD uses a temporary learned
projection into teacher width instead of pretending the two residual spaces
have identical dimensions.

Training manifests report optimizer updates, dataset passes, completion tokens,
and student-sampled tokens; `records_per_epoch=... / epochs=1` is no longer the
training-duration contract.

## 4. On-policy autoregressive KD

After teacher-forced recovery, v2 samples responses from the student itself.
For every sampled token it evaluates student and teacher log probabilities on
the same student-generated prefix and applies the K3 reverse-KL estimator:

```text
delta = log p_teacher(y_t | y_<t) - log p_student(y_t | y_<t)
K3    = exp(delta) - delta - 1
```

The on-policy stage keeps small supervised/forward-KL anchor losses, but its
main KD signal is on states the student actually visits during generation. The
default sampled response length is bounded to 4..64 new tokens.

## 5. Specialization

Only after recovery/on-policy KD passes generation gates does the final low-LR
stage focus on the Uni-GUI GUI-policy and generated screen-chat corpora. This
keeps Android specialization after basic generative behavior has recovered.

## 6. Generation gates

The initialization is evaluated before update 1, then periodically during every
stage and again at the end. The manifest records:

```text
first_token_eos_rate
empty_generation_rate
median_generated_tokens
mean_supervised_ce
mean_teacher_kl
gui_parse_rate
```

Default structural acceptance requires:

```text
first-token EOS <= 5%
empty output    <= 5%
median output   >= 8 tokens
```

Final specialization also requires at least 50% valid GUI-Owl action parsing on
the held-out GUI sample. A failed gate leaves the checkpoint and metrics for
debugging but exits non-zero, stopping the progressive cascade.

Periodic best checkpoints are retained so a later collapse does not erase a
previously stronger intermediate state.

## Recommended run order

Prepare the broad recovery JSONL first, then:

```bash
python scripts/calibrate_gui_owl_compression.py --device cuda

python scripts/run_gui_owl_progressive_distillation.py \
  --student 1.5b \
  --device cuda

python scripts/run_gui_owl_progressive_distillation.py \
  --student 1b \
  --device cuda

python scripts/run_gui_owl_progressive_distillation.py \
  --student 0.5b \
  --device cuda
```

The 1B runner refuses to start until its accepted 1.5B predecessor exists. The
0.5B runner also uses the accepted recovered 1.5B checkpoint as the structured
initialization source.

For a plumbing-only smoke run:

```bash
python scripts/run_gui_owl_progressive_distillation.py \
  --student 1.5b \
  --device cuda \
  --max-records-per-corpus 16 \
  --max-updates 2 \
  --intermediate-recovery-updates 2
```

A two-update smoke run is not a quality run and will usually fail the generation
gate. Use it to expose loading, tensor-shape, processor, and CUDA problems; do
not export its checkpoint as a model candidate.

## Docker Compose

On an Ubuntu host with Docker + NVIDIA Container Toolkit:

```bash
docker compose run --rm calibrate-student

STUDENT_SIZE=1.5b docker compose run --rm progressive-student
STUDENT_SIZE=1b   docker compose run --rm progressive-student
STUDENT_SIZE=0.5b docker compose run --rm progressive-student
```

The Compose GPU reservation now requests one GPU rather than all GPUs on a
multi-GPU host. On Vast.ai, where the workload is already running inside the
provider container, run the Python commands directly instead of Docker-in-
Docker.

## Outputs

v2 outputs are deliberately separated from the collapsed v1 checkpoints:

```text
artifacts/students/v2/gui-owl-1.5b-distilled/
artifacts/students/v2/gui-owl-1b-distilled/
artifacts/students/v2/gui-owl-0.5b-distilled/
```

Every final directory must contain `distillation-manifest.json` with:

```json
{"accepted": true}
```

before GGUF conversion should be attempted. Export integrity protects transfer
and conversion correctness; it does not replace the generation gate.

CI validates Python syntax, dependency-light pruning/gate contracts, shell
syntax, and Docker Compose configuration. It does not download the 2B model,
run calibration, execute CUDA training, or prove the structured 0.5B tensor map
against live model weights; the first bounded GPU build remains a required
runtime validation.
