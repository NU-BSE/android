# Stage 2 runbook: episode context

Stage 2 of `qwen35_08b_training_plan.md`. The plan reserved this stage for
grounding-targeted supervision. **That is cancelled.** Grounding does not need a
stage; two action classes need one, and they need data the corpus does not
currently carry.

## Why the planned Stage 2 is cancelled

The plan's Stage 2 was written before the baseline was read per record. Scoring
the 0.8B and the adapter-merged 2B teacher on the **same 192 records** and the
same annotated boxes:

| in-box rate | Qwen3.5-0.8B | GUI-Owl-2B | gap |
|---|---|---|---|
| all coordinate records | 0.506 | 0.561 | 5.5 pts |
| **click records only** | **0.643** (83/129) | **0.682** (88/129) | **3.9 pts** |
| swipe records | 0.000 (0/30) | 0.133 (4/30) | 13.3 pts |
| long_press records | 0.000 (0/5) | 0.000 (0/5) | 0 |

On click — the only action where both models are attempting the same thing —
the untrained 0.8B is 3.9 points behind a teacher that was fine-tuned on this
corpus. At n=129 the 95% half-width is ±0.083. **The gap is not measurable.**

The whole 5.5-point aggregate gap comes from swipe, and on swipe records the
0.8B is not missing the box, it is emitting the wrong action and being scored on
where its click landed. That is an action-identity failure wearing a grounding
failure's clothes. Spending a stage on coordinate supervision would have been
optimising the half of the model that already works — and, given that supervised
coordinates on this corpus are what pulled every pruned student to the corpus
median, it would have been the risky half to touch.

`target_bbox` keeps its role as the *metric* (`gui_point_in_box_rate`, added in
Stage 1). It does not become a training signal.

## What actually needs a stage

Both models score **exactly zero** on the same two classes:

| target | teacher | student |
|---|---|---|
| `mobile.system_button` | 0 / 10 | 0 / 10 |
| `mobile.answer` | 0 / 9 | 0 / 9 |

The teacher's other classes are healthy — click 0.860, swipe 0.867, type 0.875,
wait 1.000 — so this is not a weak teacher. It was fine-tuned on 175
`system_button` and 88 `answer` records in this corpus's own training split and
still never emits either. Its substitutions are revealing: for
`system_button(Back)` it emits a scroll or a click; for `answer` it emits a click
on the element holding the answer.

This is not a scoring artifact. `compare_gui_outputs` matches both classes
exactly when they are emitted, verified directly.

The two classes have one thing in common: **they depend on episode state, and a
record carries none.** `answer` means the goal's question is now satisfied;
`Back` means this screen is a dead end. One screenshot plus a goal determines
neither. The interaction contract already provides for this —
`project_gui_owl_train.py` keeps recent turns in full and compresses older ones
to their visible `Action:` conclusions — but the SFT corpus is single-step and
throws that half away. 91.3% of records have prior steps sitting unused in the
same episode.

`scripts/add_gui_policy_history.py` restores the compressed half: prior
conclusions of the same episode, in order, as text, inserted before
`Current screen:`. Screenshots are not duplicated, so a record grows by a median
**81 text tokens against ~2,025 image tokens** — 4% overhead. The addition
survives `normalize_gui_messages`, so it reaches the model under the shipping
contract. Because the splits are episode-disjoint, history never crosses the
train/holdout boundary.

A worked example, a `system_button(Home)` record the teacher gets wrong:

```
Task: Activate Google Assistant, switch to the text input mode, and type a
      command to set an alarm for 7 AM.
Steps so far:
1. ... I tapped the "Sign In" button again to retry the process.
2. The device is on the "Checking info..." screen ..., so I waited ...
3. The device is still on the "Checking info..." screen, so I waited ...
4. I waited for 5 seconds because the device is still on ...
Current screen: Couldn
```

Target: press Home and start over. Without the history the model sees a truncated
error string; with it, four failed waits make the decision obvious.

## The pre-check, and why it comes first

Adding history is a corpus rewrite and a retrain. Before paying for either,
measure whether the signal is the thing that is missing, on the model that
already knows the format: run the **teacher, zero-shot, unchanged**, over the
same records with and without history.

```bash
python scripts/add_gui_policy_history.py \
  --corpus <corpus with local image paths> \
  --output artifacts/training/probe-nohist.jsonl \
  --history-steps 0 --only-ids artifacts/diagnostics/stage2/probe-ids.json
python scripts/add_gui_policy_history.py \
  --corpus <corpus with local image paths> \
  --output artifacts/training/probe-hist.jsonl \
  --history-steps 4 --only-ids artifacts/diagnostics/stage2/probe-ids.json

for c in nohist hist; do
  python scripts/evaluate_gui_owl_policy_production.py \
    --checkpoint artifacts/teachers/gui-owl-2b-merged \
    --corpus artifacts/training/probe-$c.jsonl --device cuda \
    --holdout-fraction 1.0 --max-records 34 --max-new-tokens 64 --seed 0 \
    --output artifacts/diagnostics/stage2/teacher-$c.jsonl
done
```

The probe set is every `system_button` and `answer` record in the baseline slice
(19) plus 15 click records as a control, so a gain on the target classes can be
read against any loss on the class that already works. 64 new tokens is enough:
the teacher's median completion on this corpus is 43.

**Decision rule.** If the teacher's combined recall on the 19 target-class
records rises above zero with history, the signal was missing. If it stays at
zero, the signal is not in the compressed history either.

## Probe result: weak positive for `system_button`, null for `answer`

Run on the merged teacher, greedy, 34 records, MPS.
Artifacts in `artifacts/diagnostics/stage2/`.

| | no history | with history |
|---|---|---|
| `system_button` recall | 0.000 (0/10) | **0.100 (1/10)** |
| `answer` recall | 0.000 (0/9) | 0.000 (0/9) |
| click recall (control) | 0.867 (13/15) | 0.933 (14/15) |
| action macro recall | 0.289 | 0.344 |
| point-in-box | 0.733 | 0.800 |
| parse | 1.000 | 0.971 |
| action types emitted | 4 | **6** |

Per record: **2 flipped to correct, 0 lost.** The most informative line is not
the recall column but the last one. Without history the teacher emitted
`system_button` **zero times in 34 records**; with history it emitted it once,
on a record whose target was `system_button`. It also emitted `long_press` for
the first time. History changes the action distribution, which is the mechanism
this stage exists to trigger, and it costs nothing: the control class went up,
grounding went up, and one record stopped parsing.

**But 1 of 10 is not a result.** Two discordant pairs, both favourable, is
p=0.5 under McNemar. This is a direction, not a decision.

**So the first thing to run on the GPU is the same probe at full power.** The
settings corpus holds 175 `system_button` and 88 `answer` records against the 19
used here. Scoring even 100 of them is a few GPU-minutes and is the difference
between an effect and a coin flip. Do not rewrite the corpus and retrain on 2
flips.

```bash
python - <<'EOF'
import json, re
out = []
for line in open("artifacts/training/split-all/train.jsonl", encoding="utf-8"):
    r = json.loads(line)
    a = r["messages"][-1]["content"]
    a = a if isinstance(a, str) else "".join(p.get("text", "") for p in a)
    if re.search(r'"mobile\.(system_button|answer)"', a):
        out.append(r["id"])
json.dump(out[:150], open("artifacts/diagnostics/stage2/probe-ids-full.json", "w"))
print(len(out), "target-class records available")
EOF
```

Then the two `add_gui_policy_history.py` + evaluator runs above with
`--only-ids artifacts/diagnostics/stage2/probe-ids-full.json`.

## `answer` moves to Stage 3

Two independent measurements say `answer` is not a context problem:

1. History left it at 0/9. Its substitutions did not even change — click on five,
   swipe on three, and one new parse failure.
2. **90.9% of `answer` records (80 of 88) are the final surviving step of their
   episode.** `answer` is a termination decision. Its goal text is
   question-shaped 73.9% of the time against 30.1% for every other class, so the
   cue exists — but "stop here" cannot be learned from a corpus in which 96.8%
   of steps are "continue" and the only feature distinguishing the last one is
   that no further step exists.

Terminal verification is exactly the signal Stage 3 supplies. Exclude `answer`
from the Stage 2 gate and carry it forward with this measurement as the reason.
The Stage 2 target is therefore `system_button` alone, and the macro-recall gate
is over six reachable classes, not seven.

## If the full-power probe confirms it


```bash
python scripts/add_gui_policy_history.py \
  --corpus artifacts/training/split-all/train.jsonl \
  --output artifacts/training/split-all/train-history.jsonl --history-steps 4
python scripts/add_gui_policy_history.py \
  --corpus artifacts/training/split-all/holdout.jsonl \
  --output artifacts/training/split-all/holdout-history.jsonl --history-steps 4
```

Retrain with the Stage 1 winner's settings on `train-history.jsonl` and score on
`holdout-history.jsonl`. Report against the Stage 1 checkpoint on the *same*
holdout format — a model trained with history must not be compared to one
evaluated without it.

Gate: `gui_action_macro_recall` above the Stage 1 result with at least one of the
two target classes above 0.20 recall, and both grounding floors intact.

## A note for the size tier, not for this stage

While measuring, one number is worth recording. The GUI corpus uses **3,041
distinct token ids of 248,320** — 1.22% of the vocabulary. The tied embedding is
254M parameters, **29.8% of the 853M model**. Trimming to a frequency-ranked
32,768 tokens over the GUI and chat corpora would take the model to ~632M, a 26%
reduction, without touching a transformer block — the one axis the findings doc
identified as carrying no grounding.

This is not Stage 2 work. The 0.8B already meets its size target, and vocabulary
surgery rewrites the tied `lm_head`, so generation would have to be re-measured
from scratch. Record it for the 0.5B tier and leave it alone here.
