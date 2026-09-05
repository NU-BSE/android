# Training plan: Qwen3.5-0.8B, no pruning

`gui_owl_grounding_findings.md` closes the compression program: depth and width
pruning both delete visual grounding before any training, and every
prune-then-recover run converged to the constant-guess floor. The same document
measures the alternative — `Qwen/Qwen3.5-0.8B`, zero GUI training, lands within
a few points of the production gates. This plan starts there.

**The reframe: the 2B stops being a teacher and becomes a reference.** Nothing
carves a student any more. There is no build, no calibration manifest, no depth
schedule, no layer selection. The backbone is downloaded intact and trained.

## What the backbone actually is

Measured from the checkpoint, not from the model card.

| | GUI-Owl-2B (teacher) | Qwen3.5-0.8B |
|---|---|---|
| loaded parameters | 2.13B | **853M** |
| text layers | 28 uniform full-attention | 24 hybrid: 18 linear-attn + 6 full-attn (every 4th) |
| hidden / head_dim | 2048 / 128 | 1024 / **256** (decoupled) |
| attention heads | 16 q / 8 kv | 8 q / **2 kv**, `attn_output_gate: true` |
| FFN | 6144 | 3584 |
| vocab | 151,936 | **248,320** |
| vision tower | 24 blocks, deepstack taps 5/11/17 | 12 blocks, **no deepstack** |
| tied embeddings | yes | yes |

`AutoModelForImageTextToText` resolves it to `Qwen3_5ForConditionalGeneration`
on transformers 5.14.1. The published 873M figure counts a multi-token-prediction
head (`mtp.*`, 15 tensors) that this class does not load; 853M is what trains.

### Consequence 1: no logit-level distillation from the 2B is possible

Different tokenizers and a 248,320 vs 151,936 vocabulary. Forward KL, K3 reverse
KL and hidden-state MSE — every loss term in `scripts/distill_gui_owl_student.py`
except the supervised one — are undefined across this pair. The teacher can only
supervise **at the text level**: it relabels a student rollout with a corrected
`Action:` + `<tool_call>` string and the student does sequence learning on that.
That is on-policy distillation in the LiteGUI sense, and `OnPolicyDistiller`
already implements it. Nothing else in the distillation stack survives, and
nothing else is needed.

### Consequence 2: the current LoRA target list misses 75% of the stack

`config/gui_owl_dual_lora_data.json` targets
`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`. On this backbone the
18 linear-attention layers carry no module by those names — they carry
`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `conv1d`, `out_proj`.
Verified by attaching PEFT 0.20 to the real checkpoint:

| target list | adapted modules | trainable | linear_attn coverage |
|---|---|---|---|
| current | 96 | 12.8M | **0 of 18 layers** |
| + `in_proj_qkv,in_proj_z,out_proj` | 150 | 20.4M | 18 of 18 |

Adapt the two input projections and the output projection; leave `in_proj_a`,
`in_proj_b`, `conv1d`, `A_log` and `dt_bias` alone — those are the gating and
decay parameters of the recurrence, and a low-rank perturbation of a decay rate
is not a low-rank perturbation of a function.

### Consequence 3: the GPU needs two kernel packages

Loading the model prints: *the fast path is not available … falling back to
torch implementation*, for `flash-linear-attention` and `causal-conv1d`. The
fallback evaluates the linear-attention recurrence step by step in Python. With
~2.5k image tokens per screenshot that is the difference between a viable run
and an unviable one. Install both before measuring anything.

## Baseline to beat

From `artifacts/diagnostics/backbone/`, 192-record Android holdout, shipping
contract prompt, greedy.

| metric | 0.8B now | gate | gap |
|---|---|---|---|
| parse | 0.875 | 0.70 | passes |
| action name match | 0.635 | 0.65 | −0.015 |
| within50 | 0.354 | 0.38 | −0.026 |
| median coord error | 201.1 | ≤200 | −1.1 |
| action macro recall | **0.172** | chance 0.143 | see below |
| point-in-box (contract prompt) | **50.6%** | floor 22.0% | — |

The last two rows are re-measurements added while writing Stage 1, and they
change the reading of the first four. Per record, the 0.8B emits `click` for 165
of 192, scoring 0.922 recall on click and 0.000 on system_button, answer,
long_press and wait. Its 0.635 action match is the constant-click predictor,
below that predictor's own 0.672 on this slice — and the 0.65 gate sits between
them. Meanwhile its points land inside the annotated target element 50.6% of the
time, against the 35.4% `within50` reports.

So the deficit is not spread across the metrics: **grounding is the strong half
and action identity is at chance.** Stage 1 is a coverage problem, and the
grounding numbers are floors to protect, not targets to chase. See
`stage1_contract_adaptation.md`.

## The training plan

Four stages. Each is independently gated, ordered cheapest first, and each is
allowed to stop the run.

### Stage 0 — widen the corpus (CPU, no GPU)

2,729 records is the entire GUI corpus, and every one of them is
`app: settings` — `prepare_unigui_gui_policy.py` defaults to `--app settings`.
Re-prepare without the filter before spending GPU time on 2.7k records.

```bash
python scripts/prepare_unigui_gui_policy.py --app "" \
  --output artifacts/training/gui-policy-unigui-all.jsonl
```

An empty `--app` disables the filter (`_matches_episode` short-circuits on a
falsy app). The downloader notes this pulls 9+ GB of screenshots instead of the
1.1 GB the settings slice occupies, so run it where the disk is — the last two
production runs died of ENOSPC.

Freeze the evaluation holdout **now**, from the settings-only corpus with the
production evaluator's own shuffle, so every number below stays comparable to
the baseline table and to every measurement in the findings doc.

### Stage 1 — contract adaptation, with the coordinate question settled by measurement

Purpose: action macro recall 0.172 → ≥0.40 without losing aim; parse 0.875 → ~1.0. LiteGUI finds SFT *degrades*
small GUI agents and this pipeline reached the same conclusion from the other
side, so this stage is deliberately small: LoRA r=32, lr 1e-4 with warmup,
≤400 updates, coordinates on the 0..1000 scale.

The open risk is that supervised coordinate tokens pull the model toward the
corpus median — the exact attractor that ate every pruned student. Run it as a
two-arm ablation at 300 updates rather than guessing, because the findings doc
records two prompt choices that were optimised against the teacher and shipped
without a student measurement, and both failed:

- **arm A** — full supervision, coordinate-token loss weighted 3×.
- **arm B** — coordinate *value* tokens masked out of the loss; the structural
  tokens (`"coordinate"`, brackets, commas) stay supervised. Format is taught,
  aim is left to the pretrained weights.

Gate both on macro recall against the frozen holdout, keep the winner, keep the
loser's diagnostics as the evidence. Promotion requires parse ≥0.95, macro
recall ≥0.40, and both `gui_point_in_box_rate` and within50 no worse than
baseline minus one CI half-width. Full protocol in
`stage1_contract_adaptation.md`.

### Stage 2 — episode context (was: grounding-targeted supervision)

**Redirected.** Grounding does not need a stage. On click records — the only
action where the 0.8B and the teacher attempt the same thing — the untrained
0.8B lands 0.643 of its points inside the annotated element against the
fine-tuned teacher's 0.682, a 3.9-point gap at n=129 where the CI half-width is
0.083. The whole 5.5-point aggregate gap is swipe records, where the 0.8B is
emitting the wrong action and being scored on where its click landed.

`target_bbox` stays a metric (`gui_point_in_box_rate`, added in Stage 1) and does
not become a training signal.

What needs a stage is the two classes on which **both** models score exactly
zero: `mobile.system_button` (0/10 each) and `mobile.answer` (0/9 each). Both
depend on episode state — "answer" means the goal is now satisfied, "Back" means
this screen is a dead end — and an SFT record carries none, though 91.3% of
records have prior steps sitting unused in the same episode. Stage 2 restores
the compressed history the interaction contract already specifies, at a median
81 text tokens against ~2,025 image tokens.

The zero-shot teacher probe has been run (34 records, `artifacts/diagnostics/stage2/`).
History moved `system_button` from 0/10 to 1/10, made the teacher emit
`system_button` and `long_press` for the first time in 34 records, flipped 2
records to correct and lost none, and raised both the click control and
point-in-box. It left `answer` at 0/9. Two favourable discordant pairs is a
direction, not a decision, so the first GPU job is the same probe over ~150
target-class records rather than 19.

`answer` is reassigned to Stage 3: 80 of its 88 records are the final surviving
step of their episode, so it is a termination decision, and terminal
verification is Stage 3's signal, not a context feature. Full protocol in
`stage2_episode_context.md`.

### Stage 3 — on-policy on the emulator (the LiteGUI direction)

Student samples on a live emulator through `project_lab.AdbSettingsEnvironment`;
reward is the checkpoint-prefix progress term plus terminal verification; a
local reference model grades what the environment cannot.

The loop was run end to end before it was planned, and three of its four links
were broken: the reward function scored the untouched Settings homepage as the
forbidden search screen and penalised the first action of every episode; the
rollout path carried a private, stale copy of the contract prompt that made the
merged 2B copy the action-space alternation as its tool name; and no usable
teacher existed. All three are fixed and measured on the emulator. `answer` is
**not** rescued here — the environment returns reward 0.0 and `verified=False`
for language turns and delegates to a teacher that scores 0/9 on the class.
Full protocol and the open items in `stage3_on_policy_emulator.md`.

## Code work

Small, and mostly deletion by omission.

| file | change |
|---|---|
| `project_gui_owl_train.py:350,385` | `Qwen3VLForConditionalGeneration` → `AutoModelForImageTextToText`; `DEFAULT_MODEL_ID` from config |
| `scripts/train_gui_owl_adapter.py:113,132` | same swap |
| `config/gui_owl_dual_lora_data.json` | Qwen3.5 target-module list (above) |
| new `config/qwen35_backbone.json` | backbone id, LoRA spec, stage weights with all KD terms zeroed |
| gate config | add `gui_point_in_box_rate_min` |
| loss | coordinate-token masking / weighting utility |

`scripts/evaluate_gui_owl_policy_checkpoint.py` already resolves the model class
by architecture and already accepts a Hugging Face repo id, so evaluation needs
nothing.

Not used at all, and not to be revived: `build_gui_owl_student.py`,
`calibrate_gui_owl_compression.py`, `normalize_gui_owl_calibration.py`,
`sweep_gui_owl_width.py`, `ablate_gui_owl_text_layers.py`,
`run_gui_owl_1_5b_production.py`.

## GPU sizing

Stages 1 and 2 need one 24 GB card and no teacher: 1.7 GB of bf16 weights,
20.4M LoRA parameters, ~245 MB of AdamW state, gradient checkpointing on,
batch 1 with accumulation. Activations are dominated by image tokens — the
processor's default `longest_edge` is 16.7M pixels (16,384 image tokens); cap it
so a 1080×1920 screenshot costs ~2.5k tokens and the budget is predictable.
Stage 3 adds the 2B at ~4.3 GB bf16, which the same card holds, or the
relabelling runs as a cached offline pass.

Throughput is not estimated here on purpose: it depends almost entirely on
whether `fla` and `causal-conv1d` are installed, so measure 20 updates first and
size the run from that.

## Emulator

Connected and verified on this host: AVD `pixel_2` booted as `emulator-5554`,
API 29, which is what profile `aosp10-low-phone` in `config/emulator_matrix.json`
declares — `project_lab` refuses a device whose SDK does not match its profile,
and it refuses non-emulators outright, so the physical phone also attached to
this host can never be driven.

```bash
./android_drlqn_avd_matrix_apple_silicon.sh boot aosp10-low-phone
export ANDROID_DRLQN_ADB_SERIAL_AOSP10_LOW_PHONE=emulator-5554
export ANDROID_DRLQN_ALLOW_EMULATOR_INPUT=1
```

Verified end to end: `am start com.android.settings/.Settings` followed by
`exec-out screencap` returns a 1080x1920 PNG and `dumpsys window` reports the
Settings activity in focus, which is the whole of `_observe`'s adb surface.
(The matrix declares this profile as 720x1280; the AVD actually reports
1080x1920. `_display_size` reads the device, so nothing is broken by that, but
the declared metadata is wrong.)

Only this one AVD exists; the other eight profiles in the matrix report MISSING.
One API-29 profile is a narrower distribution than the corpus, so widen the
matrix with `prepare-stock` before treating stage-3 results as representative.
