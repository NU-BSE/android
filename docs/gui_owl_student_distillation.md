# GUI-Owl 2B -> 1.5B / 1B / 0.5B student distillation

This pipeline converts the trained GUI-Owl 2B teacher into full compact Qwen3-VL
student checkpoints. It is separate from LoRA training: the students have
smaller architectures and their complete weights are trainable and saved.

## Inputs

The teacher behavior comes from the same base model and two adapters produced by
the dual-LoRA pipeline:

```text
mPLUG/GUI-Owl-1.5-2B-Instruct
├── gui_policy adapter <- Uni-GUI-OpenMobile trajectories
└── chat adapter       <- Qwen3.7-Plus screen-chat data
```

The student receives both corpora and becomes one unified checkpoint. During KD,
GUI records switch the teacher to `gui_policy`; chat records switch it to
`chat`. The student itself has no runtime adapter switch.

The architecture plan lives in `config/gui_owl_students.json`.

## Student architectures

The published GUI-Owl 2B config has a 28-layer, 2048-wide language model and a
24-layer, 1024-wide vision tower. The 1B/1.5B students keep the tokenizer,
special-token contract, language width, and complete vision tower and reduce
language depth. The 0.5B student must narrow both modalities; retaining the full
2048-wide embedding and vision stack cannot reach the 0.5B class.

```text
1.5B: text 2048 / FFN 6144 / 16 blocks, original 24-block vision tower
1.0B: text 2048 / FFN 6144 /  6 blocks, original 24-block vision tower
0.5B: text 1024 / FFN 3072 / 16 blocks, 12-block 768-wide vision tower
```

Exact parameter counts are calculated after Transformers instantiates each
student. The builder fails when a candidate is outside its configured tolerance
unless `--allow-parameter-miss` is explicitly supplied. This makes the size
label auditable rather than relying on a hand-estimated parameter formula.

## Stage A: initialize compact students from the 2B base

Language blocks are copied from evenly spaced teacher layers. The 1B map is:

```text
0, 5, 11, 16, 22, 27
```

For tensors whose dimensions are narrowed in the 0.5B model, the builder copies
the common leading sub-tensor and leaves no dependency on the original model at
runtime. Processor/tokenizer files are copied from the teacher.

Build 1B first:

```bash
python scripts/build_gui_owl_student.py --student 1b
```

Then 1.5B:

```bash
python scripts/build_gui_owl_student.py --student 1.5b
```

And initialize 0.5B from the original 2B architecture:

```bash
python scripts/build_gui_owl_student.py --student 0.5b
```

Outputs default to:

```text
artifacts/students/gui-owl-1.5b-init/
artifacts/students/gui-owl-1b-init/
artifacts/students/gui-owl-0.5b-init/
```

Each directory contains `student-build-manifest.json` with exact parameters,
layer routing, initialization coverage, dtype, and source teacher.

## Stage B: multimodal knowledge distillation

For every record, the student and teacher see the same tokenized prompt and, for
GUI examples, the same screenshot. Loss is applied only over the assistant
completion window:

```text
L = w_corpus * (
      alpha * L_supervised
    + beta  * T^2 KL(p_teacher^T || p_student^T)
    + gamma * L_hidden
)
```

Defaults:

```text
T = 2.0
alpha = 0.5
beta = 1.0
gamma = 0.1 for equal-width 1B/1.5B students
gamma = 0.0 for the narrowed 0.5B student
```

For 1B and 1.5B, student hidden states are matched against the evenly routed 2B
teacher layers. The unchanged vision tower is frozen during KD, reducing memory
and preventing drift. The 0.5B model trains its smaller vision tower and learns
visual behavior through the supervised and teacher-logit objectives.

The loader normalizes per-record corpus weights so a much larger generated chat
JSONL does not automatically swamp the smaller GUI corpus.

Distill 1B from the trained 2B + adapters:

```bash
python scripts/distill_gui_owl_student.py \
  --student 1b \
  --device cuda
```

Distill 1.5B the same way:

```bash
python scripts/distill_gui_owl_student.py \
  --student 1.5b \
  --device cuda
```

For a low-cost smoke test before a full rental:

```bash
python scripts/distill_gui_owl_student.py \
  --student 1b \
  --device cuda \
  --max-records-per-corpus 64 \
  --output artifacts/students/gui-owl-1b-smoke
```

## Recommended 0.5B cascade

Do not use direct 2B -> 0.5B as the default final run. First distill the 1B
student, then use that unified 1B checkpoint as the logit teacher for 0.5B:

```text
2B + gui/chat adapters
        |
        +--> 1.5B
        |
        +--> 1B unified student
               |
               +--> 0.5B
```

The 0.5B architecture is still initialized from the 2B base so its 16 text
blocks can be routed across all 28 original blocks. Its KD teacher can then be
the already-distilled 1B checkpoint:

```bash
python scripts/distill_gui_owl_student.py \
  --student 0.5b \
  --teacher-model artifacts/students/gui-owl-1b-distilled \
  --teacher-unified \
  --device cuda
```

Hidden-state KD is disabled automatically for this width-changing stage unless
explicitly overridden.

## Docker Compose / Vast.ai

The stacked Uni-GUI PR already provides the CUDA training image. The distillation
PR adds generic Compose services controlled through `.env`:

```bash
STUDENT_SIZE=1b docker compose run --rm build-student
STUDENT_SIZE=1b docker compose run --rm distill-student
```

For a bounded KD smoke run:

```bash
STUDENT_SIZE=1b DISTILL_MAX_RECORDS=64 \
  docker compose run --rm distill-student
```

After `gui-owl-1b-distilled` exists, the recommended 0.5B cascade is:

```bash
STUDENT_SIZE=0.5b \
DISTILL_TEACHER_MODEL=artifacts/students/gui-owl-1b-distilled \
DISTILL_TEACHER_UNIFIED=1 \
  docker compose run --rm distill-student
```

## Outputs and deployment boundary

Distillation outputs are complete Hugging Face Qwen3-VL checkpoints, not PEFT
adapters:

```text
artifacts/students/gui-owl-1.5b-distilled/
artifacts/students/gui-owl-1b-distilled/
artifacts/students/gui-owl-0.5b-distilled/
```

Each includes `distillation-manifest.json` with teacher source, objectives,
record/update counts, and mean loss components. No API credentials are written.

GGUF/mmproj conversion and llama.rn packaging are intentionally a subsequent
stage. First gate each student on held-out GUI action validity, navigation
success, and chat quality; then export only students that pass those gates.

No model weights, generated checkpoints, credentials, or external datasets are
committed by this pipeline.
