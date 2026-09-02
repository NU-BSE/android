# Visual grounding does not survive depth pruning

The compact GUI-Owl students emit well-formed tool calls and cannot navigate.
This documents why, because the cause is not where the pipeline was looking and
none of the existing gates can see it.

**Finding: removing text layers destroys visual grounding before any training
happens. One layer costs 59% of it; four costs all of it. Syntax survives
intact throughout, which is why every acceptance gate passed.**

All measurements below use `compare_gui_outputs` — the same metric the
production gates use — over held-out GUI-policy records under the shipping
contract prompt, unless stated otherwise. `within50` is the share of
coordinate-bearing targets whose predicted point lands within 50 units on the
0..1000 scale.

## The depth sweep

Every row is pruned from the same adapter-merged 2B teacher. No training of any
kind: build, then evaluate. Layers are dropped in ascending calibration
saliency, so each row nests inside the one above it.

| depth | dropped | median err | within50 | 95% CI | action | parse |
|---|---|---|---|---|---|---|
| 28 | none | 128.0 | **46.5%** | [40.5, 52.6] | 0.766 | 1.000 |
| 27 | {12} | 222.5 | **19.2%** | [14.3, 25.4] | 0.688 | 0.995 |
| 26 | {12,13} | 222.5 | **16.2%** | [11.6, 22.0] | 0.682 | 1.000 |
| 25 | {12,13,14} | 309.7 | **4.2%** | [2.1, 8.0] | 0.682 | 1.000 |
| 24 | {7,12,13,14} | 354.8 | **0.0%** | [0.0, 1.5] | 0.648 | 1.000 |
| — | *constant-guess floor* | 270.9 | *5.8%* | | | |

The decay is monotonic and crosses the constant-guess floor between depth 26
and 25. "Constant-guess floor" is what you score by ignoring the screenshot
entirely and always emitting the corpus median point (500, 645); uniform random
guessing scores 449.2px / 0.7%.

The largest single drop is the **first** layer removed. There is no plateau and
no safe depth budget: the 1.5B target needs to reach depth 20.

Identity of the dropped layers does not rescue it. A depth-24 built by dropping
{3,4,5,6} instead collapses differently but no less completely — parse 0.042,
action 0.026 — so it is depth itself, not which blocks are chosen.

## Syntax is not affected, which is why nothing caught it

Across the entire sweep `parse` stays at ~1.000 and action-name match drifts
only 0.766 → 0.648. The model keeps producing well-formed
`Action:` + `<tool_call>{...}</tool_call>` output with plausible action names
all the way down to zero grounding.

`gui_parse_rate` is a **syntax** check: it asks whether
`parse_gui_owl_output` returns non-None, never whether the action or the
coordinate is right. An earlier accepted 1.5B scored 98.4% parse on 64 unseen
records — higher than its own teacher's 93.8% — while emitting `mobile.click`
for 63 of 64 records and landing a median 340px from target, with 0 of 41
predictions within 50px. It passed every gate and could not navigate.

## How the cause was localised

Each step ruled out a hypothesis that looked plausible at the time.

1. **The student's coordinates carry no perceptual signal.** depth-24 after
   recovery: 272.6px median, 0.9% within50 — indistinguishable from the
   best-constant-guess baseline of 270.9px / 5.8%, and *below* it on within50.
2. **It is not a training-budget problem.** Across recovery's 1800 updates
   `within50` never left 0.000–0.009 and the median pinned at 279 ± 4 from
   update 100 onward. The stage reached the constant-guess optimum in its first
   100 updates and held it for the remaining 1700.
3. **It is not the contract bootstrap.** Per-chunk manifests show `within50` at
   0.0047 from chunk-1 onward — it was already gone after 100 updates. What the
   bootstrap does teach is emitting a coordinate at all (median 1414 → 349).
4. **It is not an acquisition problem.** The base teacher without its GUI
   adapter still grounds: 222.5px / 28.1%. Grounding is present in the weights
   the student is initialised from.
5. **The adapter merge is lossless.** Merged teacher 128.0px / 46.5% against
   the adapter-attached teacher's 128.3px / 45.2%, so a pruned merged model is
   a valid measurement.
6. **The prune destroys it.** Merged teacher 46.5% → merged-and-pruned-to-24
   0.0%, with no training in between.

## What this rules out

The pipeline is built on prune-then-recover. For grounding the prune is not
degradation that recovery repairs — it is deletion, and recovery converges to a
constant instead. Consequently none of these were the cause, though each was
investigated: the contract prompt, the gate thresholds, the K3 clamp, the
first-token weighting, the tail-layer protection, the recovery corpus modality,
or the training budget.

## Methodological findings worth keeping

**Saliency scoring is orthogonal to grounding.** Layer 12 is the *lowest*
scoring block in the model under `activation_energy_and_residual_saliency`
(0.111, against 0.28–0.42 mid-stack). Removing it alone costs 59% of grounding.
Every selection in this pipeline — depths 24, 20, 16 and the whole 1B schedule
— is driven by that metric. It cannot see the capability that matters most.

**A syntax gate cannot stand in for a correctness gate.** See above; the
`gui_action_name_match_rate` and coordinate gates were added for this reason.

**Prefer a proportion to a median for grounding.** The teacher's error
distribution is bimodal (p25 3px, p50 128px, p90 619px), so its median
bootstraps to a 95% CI of [28, 159]px at n=256 — 130px of slack on an identical
checkpoint. `gui_coordinate_within_50_rate` over the same sample has a CI width
of 0.13 and cannot swing an order of magnitude.

**Calibrate gates against the teacher under the prompt that ships.** The
original 0.70 / 100px gates were set from a teacher measured under a prompt
nothing uses (2.2px). Under the shipping contract the same teacher scores 0.762
and 128.3px — it failed its own students' coordinate gate, so no student could
pass without beating the model it is distilled from.

**Teacher ablation does not predict student behaviour.** Prompt variants were
optimised against the teacher twice and shipped twice; both failed on the
student. The teacher already knows the output format, so removing the worked
example is free for it and fatal for a student learning the format:

| prompt | teacher median err | student contract bootstrap |
|---|---|---|
| no example | 4.0px | 0% open after 200 updates |
| coordinate-free examples | 13.0px | peaked 76.2%, then declined |
| one click coordinate | 100.0px | not run |
| click + swipe coordinates (shipping) | 128.0px | 95.3% at chunk 6 — the only pass |
| three varied coordinates | 234.1px | not run |

Verbatim copying is not the mechanism: the one-click variant copied its own
example point 0 times out of 86 and still scored 100px, falling back to the
corpus's own (500,800) attractor. Any coordinate in the context degrades
grounding whether or not that value is reused.

## Open directions

Depth compression and grounding are incompatible on this architecture, so the
1.5B target needs a different approach:

- **Width pruning instead of depth.** Untested for grounding; the 0.5B path
  already has the machinery. Measurable locally with this same harness.
- **Keep 28 layers, compress elsewhere.** Quantisation to hit the size target
  rather than an architecture change.
- **Reconstruct grounding explicitly.** Only if the above fail. The current
  supervised signal is 3–4 coordinate tokens out of ~40 and provably converges
  to the corpus median.

Whichever is chosen, the saliency metric should be validated against grounding
before it drives another selection.

## Reproducing

All of the above runs on an M-series Mac in float32 on MPS, no GPU required.
MPS and CPU were verified to agree exactly on the probe.

```bash
# merge the GUI adapter into the teacher (grounding lives partly in it)
python - <<'EOF'
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
m = Qwen3VLForConditionalGeneration.from_pretrained("mPLUG/GUI-Owl-1.5-2B-Instruct", dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, "artifacts/adapters/gui_policy/gui_policy").merge_and_unload()
m.save_pretrained("artifacts/teachers/gui-owl-2b-merged", safe_serialization=True)
AutoProcessor.from_pretrained("mPLUG/GUI-Owl-1.5-2B-Instruct").save_pretrained("artifacts/teachers/gui-owl-2b-merged")
EOF

# prune it to any depth, then measure grounding under the shipping prompt
python scripts/build_gui_owl_student.py --student 1.5b \
  --source-model artifacts/teachers/gui-owl-2b-merged --target-text-layers 27 \
  --output artifacts/students/probe-depth27 --allow-parameter-miss --overwrite

python scripts/evaluate_gui_owl_policy_production.py \
  --checkpoint artifacts/students/probe-depth27 \
  --corpus artifacts/training/gui-policy-unigui-openmobile.jsonl \
  --device mps --holdout-fraction 0.2 --max-records 192 --max-new-tokens 96 \
  --output artifacts/diagnostics/probe-depth27.jsonl
```

Corpus image paths are recorded as `/workspace/android/...`; rewrite the prefix
to the repository root before evaluating off the training host.

Supporting tools added for this investigation:

- `scripts/probe_gui_owl_tool_call_rank.py` — teacher-forced rank, probability,
  top-1 mass and entropy for a token at its true target position.
- `scripts/ablate_gui_owl_text_layers.py` — measures a depth step by removing
  exactly the layers it would remove, without training.
- `scripts/diagnose_gui_tool_call_failures.py` — buckets parse failures by the
  parser's own decision points, using `parse_gui_owl_output` as the oracle.
