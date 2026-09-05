# Stage 1 runbook: contract adaptation

Stage 1 of `qwen35_08b_training_plan.md`. Two short LoRA runs on the widened
corpus, an ablation between them, and a promotion decision. Nothing here is
worth more than a few GPU-hours; the point is to buy action coverage without
paying for it in grounding.

Stage 0 must be complete. Stage 1 reads `split-all/train.jsonl` and is measured
on both holdouts.

## What the baseline actually is

The plan doc said the 0.8B needed "3 → 5 action types". That understated it.
Re-reading `artifacts/diagnostics/backbone/contract-qwen35-08b.jsonl` per record:

| target action | correct / total | recall |
|---|---|---|
| click | 119 / 129 | **0.922** |
| swipe | 1 / 30 | 0.033 |
| type | 2 / 8 | 0.250 |
| system_button | 0 / 10 | 0.000 |
| answer | 0 / 9 | 0.000 |
| long_press | 0 / 5 | 0.000 |
| wait | 0 / 1 | 0.000 |

It emitted `click` for 165 of 192 records, `swipe` 3 times, `type` twice, and
nothing at all for 22. **The backbone has one action.**

Two consequences.

**The action gate is below its own majority-class floor.** The episode-disjoint
holdout is 65.3% `mobile.click`; the 192-record slice is 67.2%. `SEMANTIC_ACTION_MIN`
is 0.65. A model that answers `mobile.click` to everything scores 0.653–0.672 and
passes. The 0.8B's 0.635 is *below* the constant predictor, and the gate sits
between them. This is the same defect the findings doc documents in
`gui_parse_rate` — a gate a degenerate model clears — now in the action gate.
Macro-averaged recall separates them: the 0.8B scores **0.172** over seven
classes against a uniform-chance 0.143.

**Grounding is better than `within50` reports.** Scoring the same generations
against the annotated `target_bbox`, the 0.8B lands **50.6%** of its points
inside the target element (83/164) where `within50` says 35.4%. The annotated
coordinate is the box centre — 100% of 2,335 annotated records have their label
inside their own box, median offset 0.5 units — so `target_bbox` contributes a
*tolerance*, not a different target, and a 50px radius against a median 224x64
box is simply the wrong tolerance.

`scripts/evaluate_gui_owl_policy_checkpoint.py` now emits
`gui_action_macro_recall`, `gui_action_class_recall`,
`gui_action_majority_class_rate`, `gui_point_in_box_rate` and `gui_box_records`
alongside the existing fields, so a pass can be read rather than trusted.

## Objective

Raise action coverage. Protect aim. In numbers, from the Stage 0 re-baseline:

| metric | baseline | Stage 1 target |
|---|---|---|
| `gui_action_macro_recall` | 0.172 | **≥ 0.40** |
| `gui_parse_rate` | 0.875 | ≥ 0.95 |
| `gui_point_in_box_rate` | 0.506 | ≥ baseline − one CI half-width |
| `gui_coordinate_within_50_rate` | 0.354 | ≥ baseline − one CI half-width |
| `gui_action_majority_class_rate` | 0.653 | reported, never gated on |

The two grounding rows are *floors, not targets* — Stage 1 is not asked to
improve aim, only to avoid trading it away. Set them from the Stage 0 numbers on
the episode-disjoint holdout, not from the values above, which come from the
leaked 192-record slice. The half-width is `1.96 * sqrt(p(1-p)/n)`; at
p=0.506, n=512 that is 0.043, so the floor would be 0.463.

## The ablation

The risk is named in the findings doc: supervised coordinates on this corpus are
3–4 tokens out of ~40 and provably converge to the corpus median. Two arms, same
data, same schedule, same seed:

- **arm W** — coordinate literals weighted 3x in the loss.
- **arm M** — coordinate literals excluded from the loss. Structural tokens
  (`"coordinate"`, brackets, commas) stay supervised, so the format is taught
  and aim is left to the pretrained weights.

`scripts/train_qwen35_gui_policy.py` implements both behind
`--coordinate-loss {weighted,masked}`. It locates the literals by decoding the
completion cumulatively and mapping character spans back to tokens; validated
against all 2,729 shipping records, it marks exactly the digits of `coordinate`
and `coordinate2` on 2,729 of 2,729, and leaves a record unmasked rather than
guessing if decoding is ever non-monotonic.

Given the priors — grounding already at 50.6% in-box, action identity at chance
— arm M is the one to expect. Run both anyway. This project has twice shipped a
choice validated only against the teacher.

## Running it

From `/workspace/android` on the Vast box, `PYTHONPATH=$PWD`.

### 1. Time one arm before committing to it

```bash
python scripts/train_qwen35_gui_policy.py \
  --train artifacts/training/split-all/train.jsonl \
  --output /tmp/timing --coordinate-loss masked \
  --max-updates 5 --gradient-accumulation 8 --log-interval 1 --device cuda
```

`elapsed_s` in the printed manifest over 40 records gives the per-record cost.
Multiply by 2,400 for a 300-update arm. If it is more than a few seconds per
record, `flash-linear-attention` did not install and the 18 linear-attention
layers are running the Python fallback — fix that first, it is the whole budget.

### 2. Both arms

```bash
for arm in masked weighted; do
  python scripts/train_qwen35_gui_policy.py \
    --train artifacts/training/split-all/train.jsonl \
    --output artifacts/students/qwen35-08b/stage1-$arm \
    --merged-output artifacts/students/qwen35-08b/stage1-$arm-merged \
    --coordinate-loss $arm --coordinate-weight 3.0 \
    --action-balance sqrt --lora-rank 32 --lora-alpha 64 \
    --learning-rate 1e-4 --max-updates 300 --gradient-accumulation 8 \
    --warmup-fraction 0.1 --device cuda --seed 0
done
```

LoRA r=32 over the 10 target modules reaches 150 modules and 20.4M trainable
parameters; the trainer refuses to start if it attaches to fewer. Vision is
untouched — no vision module name matches the target list, and every base weight
is frozen. `--merged-output` writes a standalone checkpoint because the
evaluator loads with `AutoModelForImageTextToText` and cannot read an adapter
directory.

`--action-balance sqrt` samples inversely to the square root of the class count,
which lifts `wait` (38 records) and `long_press` (74) without drowning the run in
them. It is a hyperparameter, not a conclusion — see step 4.

### 3. Score both arms on both holdouts

```bash
for arm in masked weighted; do
  for split in settings all; do
    python scripts/evaluate_gui_owl_policy_production.py \
      --checkpoint artifacts/students/qwen35-08b/stage1-$arm-merged \
      --corpus artifacts/training/split-$split/holdout.jsonl \
      --device cuda --holdout-fraction 1.0 --max-records 512 \
      --max-new-tokens 96 --seed 0 \
      --output artifacts/diagnostics/stage1/$arm-$split.jsonl
  done
done
```

Compare on `gui_action_macro_recall` first, then confirm neither grounding floor
is breached. Read `gui_action_class_recall` directly: an arm that reaches 0.40
macro by learning `swipe` alone is not the same result as one that reaches it
across five classes, and the scalar cannot tell them apart.

If you passed `--max-image-pixels` during training, pass the same value here.
The trainer bakes the cap into the saved processor, so a checkpoint trained with
a cap and evaluated without one is measured on a different input distribution
than it saw.

### 4. Sweep the sampler on the winner only

```bash
for balance in none balanced; do
  python scripts/train_qwen35_gui_policy.py \
    --train artifacts/training/split-all/train.jsonl \
    --output artifacts/students/qwen35-08b/stage1-$WINNER-$balance \
    --merged-output artifacts/students/qwen35-08b/stage1-$WINNER-$balance-merged \
    --coordinate-loss $WINNER --action-balance $balance \
    --max-updates 300 --gradient-accumulation 8 --device cuda --seed 0
done
```

Four training runs total. Do not sweep both axes at once; the corpus is small
enough that a 2x3 grid mostly measures seed noise.

## Acceptance

- [ ] `gui_action_macro_recall` ≥ 0.40 on the **all-apps** holdout, with at least
      five of seven classes above 0.20 recall
- [ ] `gui_parse_rate` ≥ 0.95
- [ ] `gui_point_in_box_rate` and `gui_coordinate_within_50_rate` both at or
      above their Stage 0 floors
- [ ] `gui_action_majority_class_rate` recorded in the same summary, so the
      macro number can be read against it
- [ ] the losing arm's diagnostics kept, not deleted — it is the evidence for
      the coordinate decision
- [ ] settings-holdout numbers reported beside the all-apps ones, so the
      distribution change is visible

If arm M wins, Stage 2 inherits a model whose coordinates were never supervised
on this corpus, and the grounding work there starts from pretrained aim rather
than from a fitted median. If arm W wins, record by how much, because it
contradicts the mechanism the findings doc established and that is worth
knowing before Stage 2 designs a coordinate objective around it.
