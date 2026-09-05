# Visual grounding does not survive depth pruning

The compact GUI-Owl students emit well-formed tool calls and cannot navigate.
This documents why, because the cause is not where the pipeline was looking and
none of the existing gates can see it.

**Finding: removing text layers destroys visual grounding before any training
happens. One layer costs 59% of it; four costs all of it. Syntax survives
intact throughout, which is why every acceptance gate passed.**

Width pruning was tried next and fails the same way. The resolution is not a
better prune: an off-the-shelf 0.87B multimodal model lands just under the
production gates with no GUI training at all, while no pruned student ever
left the constant-guess floor. Read the last two sections first if you are
here to decide what to build.

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

## Width pruning does not rescue it

The open question above was whether the loss is specific to depth. It is
not. Every row below keeps all 28 text layers and narrows one or more axes
inside the blocks, built from the same merged teacher with the production
builder (`scripts/build_gui_owl_student.py`, structured slicing, no prefix
truncation) and measured with the same harness on the first 128 records of
the same holdout (109 coordinate targets). No training anywhere.

| variant | params | vs 1.5B | parse | action | median err | within50 | 95% CI |
|---|---|---|---|---|---|---|---|
| 28 layers, unpruned | 2.13B | +42% | 1.000 | 0.742 | 142.0 | **44.0%** | [35.1, 53.4] |
| FFN 6144→4096 | 1.78B | +18% | 0.984 | 0.625 | 258.0 | **9.2%** | [5.1, 16.1] |
| FFN 6144→3072 | 1.60B | +7% | 0.945 | 0.383 | 363.1 | **0.0%** | [0.0, 3.4] |
| heads 16/8→12/6 | 2.04B | +36% | 0.000 | 0.000 | — | 0.0% | |
| FFN 4096 + heads 12/6 | 1.69B | +12% | 0.000 | 0.000 | — | 0.0% | |
| FFN 3072 + heads 12/6 | 1.51B | +1% | 0.000 | 0.000 | — | 0.0% | |
| — | *constant-guess floor* | | | | 270.9 | *5.8%* | |

MLP-neuron pruning reproduces the depth signature exactly: syntax intact,
grounding gone. A third of the FFN removed sits where dropping two or three
layers sits, just above the constant-guess floor, and the half-FFN cut that
reaches the parameter budget is at zero. Per parameter removed it is somewhat
gentler than depth (the 1.78B FFN-4096 model keeps 9% where the 1.93B depth-24
model keeps 0%), but nowhere near the 38% gate, and the decay is monotonic
with no plateau.

Attention-head pruning is worse than either. Removing two of eight KV groups
per layer leaves a model that answers a text-only prompt coherently and emits
one or two tokens then EOS as soon as a screenshot is in context, for every
record. Whatever those heads do, they do it over image tokens.

Residual-width pruning, the axis the 0.5B recipe leans on hardest, collapses
without any image at all: 2048→1792 degenerates into repetition and 2048→1536
answers with EOS. So the three width axes the 0.5B build applies together
(hidden 2048→1024, heads 16→8, FFN 6144→3072) each destroy the model on their
own before the vision tower is touched.

### The neuron-importance metric was wrong, and fixing it was necessary but not sufficient

The first width run collapsed far harder than the table above: FFN 3072 gave
121 of 128 empty generations and first-token EOS with p≈0.4, and a **random**
neuron ranking did no worse (EOS p=0.27). The calibrator ranked MLP neurons by
the energy of the `gate_proj` pre-activation, which is not what a neuron
contributes: a large negative pre-activation has high energy and zero output
through SiLU. Ranking by the gated activation that enters `down_proj` instead
keeps the model coherent at FFN 4096 and even 3072 (text-only EOS p=0.002 /
0.015 against 0.35 / 0.43), which is the difference between "dead" and the
table above. The peak-activation ranking alone was not enough (coherent at
4096, EOS at 3072); the mean gated energy is what the calibrator now records
and the builder refuses to build a text-MLP width cut from a manifest without
it. The same flaw was in the vision-MLP ranking (`linear_fc1` pre-GELU energy)
and is fixed the same way.

That fix moves generation from dead to intact. It moves grounding from 0% to
0% at the budget, because activation energy does not know which neurons the
coordinate tokens need any more than residual saliency knew which layers they
needed.

## A pretrained 0.87B backbone starts where the pruned students never arrived

Everything above asks how to carve a small grounded model out of the 2B. That
question has a cheaper answer: do not carve one, download one.

Qwen3.5 is natively multimodal at every size, with no separate `-VL` branch.
`Qwen/Qwen3.5-0.8B` is 873,438,784 parameters, and `transformers` 5.14.1 loads
it as `Qwen3_5ForConditionalGeneration`. Measured on the same 192-record
Android holdout, under the same shipping contract prompt, greedy, on MPS:

| metric | GUI-Owl-2B | Qwen3.5-0.8B | gate |
|---|---|---|---|
| parse | 1.000 | 0.875 | 0.50 |
| action name match | 0.760 | **0.635** | 0.65 |
| within50 | 0.439 | **0.354** | 0.38 |
| median coord error | 142.0 | **201.1** | 200 |
| first-token EOS | 0.000 | 0.000 | 0.05 |
| distinct action types | 5 | 3 | — |

The 2B is adapter-merged and was fine-tuned on this corpus's own training
split. The 0.8B has seen neither the corpus nor the output format. It is
nevertheless within 1.5 points of the action gate, 2.6 of the coordinate gate
and 1.1 of the median-error gate, at 41% of the parameters and zero GUI
training. Every pruned student in this document scored **0.0%** within50 after
paid recovery. Nothing in the compression program ever came within sight of
these gates; an untrained general model does.

To separate seeing from formatting, `probe_gui_grounding_backbone.py` asks for
a tap point in any form and scores it against the annotated `target_bbox`,
which the corpus already carries on 2335 of 2729 records and which no metric
was using:

| model | point in box | within 50px | within 200px | median miss |
|---|---|---|---|---|
| GUI-Owl-2B | 39.6% | 49.4% | 65.9% | 239px |
| Qwen3.5-0.8B | 32.9% | 40.2% | 50.0% | 317px |
| *best fixed point* | *22.0%* | | | |

Both sit well above the 22.0% floor a model reaches by ignoring the screenshot
and always tapping the single best point. The 0.8B emitted 151 distinct points
out of 159, with no attractor, and hit 13 records the 2B missed, so its
competence is not a subset of the teacher's. Its weakness is aim, not sight:
the gap widens from 6.7 points at the box edge to 15.9 points at 200px, so it
has the heavier tail of far misses. Its three emitted action types against the
teacher's five is a coverage gap, not a grounding one.

Two measurement notes worth keeping. The teacher scores 39.6% in-box under the
neutral probe and 56.1% under the contract prompt, so prompt framing is worth
roughly 16 points to it, and no cross-prompt comparison is meaningful.
And a 50px radius is *stricter* than the target it is checking: the median
annotated box is 224 by 64 on the 0..1000 scale, so `within50` fails clicks
that would land on the element. Prefer point-in-box.

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

Neither depth nor width compression of the text stack survives without
training on this architecture, and every prune-then-recover run so far has
recovered to the constant-guess floor. The measurement above says the target
sizes are reachable without compressing anything, so that is the first
direction and the rest are fallbacks.

- **Swap the backbone instead of building one.** Start the small students from
  `Qwen3.5-0.8B` and `Qwen3.5-2B` rather than from a pruned 2B. This is the
  only path measured to begin above the constant-guess floor. It is loader and
  config work, not modelling work, but it is not free: the builder, metrics
  and trainers all assume `qwen3_vl`. A `qwen3_5` backbone has no deepstack,
  a decoupled `head_dim` of 256, and a hybrid stack of 18 linear-attention and
  6 full-attention layers, which breaks the layer-selection contract outright.
  Nothing in the depth or width machinery survives the swap, and none of it
  needs to.
- **Close the gap with post-training, not architecture.** The remaining
  distance to the gates is a few points on three metrics. LiteGUI
  (arXiv 2605.07505) moves a comparable backbone by about that much with
  on-policy distillation and RL, and explicitly finds that SFT *degrades* a
  small GUI agent, which is the same result this pipeline reached from the
  other side.
- **Keep 28 layers and full width; compress elsewhere.** Quantisation to hit
  the size target rather than an architecture change. The 2.13B teacher at
  Q4 is ~1.3 GB, under the 1.5B-at-bf16 file the target was sized for. This is
  the fallback if the backbone swap stalls.
- **Prune what grounding does not use.** The tied embedding is 311M of the
  2.13B and covers a 151,936-token multilingual vocabulary; a vocabulary
  restricted to the tokens the GUI/chat corpora and English need would remove
  most of it without touching a single block. It is the only 15% of the model
  with no reason to carry grounding, and it applies more strongly to
  Qwen3.5-0.8B, where a 248,320-token vocabulary at hidden size 1024 is about
  29% of the parameters.
- **Grounding-targeted importance.** Rank neurons by their first-order Taylor
  contribution to the coordinate-token loss on GUI records rather than by
  activation energy. Only worth running if compression is revived.
- **Reconstruct grounding explicitly.** Last resort. The current supervised
  signal is 3–4 coordinate tokens out of ~40 and provably converges to the
  corpus median.

Whichever is chosen, validate the selection against grounding with the width
sweep, the layer ablation, or the backbone probe before it drives a paid run.

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
- `scripts/sweep_gui_owl_width.py` — builds width variants of a checkpoint
  with the production builder and measures grounding without training;
  `--selection random` is the control for the importance metric.
- `scripts/probe_gui_grounding_backbone.py` — scores raw screenshot grounding
  against `target_bbox` with a format-free prompt, so a backbone that has
  never seen the tool-call contract can still be measured. Predictions outside
  0..1000 are rescaled by the real screenshot size, and the report says how
  many needed it.

The width sweep above was run as:

```bash
python scripts/sweep_gui_owl_width.py \
  --source-model artifacts/teachers/gui-owl-2b-merged --device mps \
  --corpus <corpus with local image paths> --max-records 128 --max-new-tokens 96 \
  --variant ffn4096=text_intermediate_size:4096 \
  --variant ffn3072=text_intermediate_size:3072 \
  --variant heads12=text_attention_heads:12,text_key_value_heads:6 \
  --variant budget=text_intermediate_size:3072,text_attention_heads:12,text_key_value_heads:6
```

against a manifest re-calibrated locally with the gated-activation neuron
scores (`scripts/calibrate_gui_owl_compression.py --device mps`; the layer
scores it produces match the CUDA manifest to three decimals).

The backbone comparison was run as the three commands below. The candidate and
the teacher go through the identical probe, and the candidate additionally goes
through the production evaluator so format compliance is measured separately
from grounding. `evaluate_gui_owl_policy_checkpoint.py` now resolves the model
class by architecture rather than hardcoding Qwen3-VL, which is what lets a
`qwen3_5` checkpoint be scored under the identical prompt and metric; it also
accepts a Hugging Face repo id in place of a local directory.

```bash
python scripts/probe_gui_grounding_backbone.py --model Qwen/Qwen3.5-0.8B \
  --corpus <corpus with local image paths> --device mps \
  --max-records 192 --max-new-tokens 64 \
  --output artifacts/diagnostics/backbone/probe-qwen35-08b.jsonl

python scripts/probe_gui_grounding_backbone.py \
  --model artifacts/teachers/gui-owl-2b-merged \
  --corpus <corpus with local image paths> --device mps \
  --max-records 192 --max-new-tokens 64 \
  --output artifacts/diagnostics/backbone/probe-guiowl-2b.jsonl

python scripts/evaluate_gui_owl_policy_production.py --checkpoint Qwen/Qwen3.5-0.8B \
  --corpus <corpus with local image paths> --device mps \
  --holdout-fraction 0.2 --max-records 192 --max-new-tokens 96 \
  --output artifacts/diagnostics/backbone/contract-qwen35-08b.jsonl
```

The probe selects its holdout with the production evaluator's own shuffle, so
both cover the same records, and it counts an unparseable answer as a miss
rather than dropping it from the denominator.
