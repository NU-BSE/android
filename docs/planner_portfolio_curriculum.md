# Planner portfolio, offline curriculum, and submission hardening

This extension treats search as a portfolio rather than asking one Q-router to
imitate every programmable Duck behavior. All specialists share the same
rendered-state transition graph. They use legal lambda actions, observed visual
effects, and discovered transitions only; no specialist reads a game identifier
or environment source implementation.

## Generic specialist portfolio

`drlqn/planning/portfolio.py` provides five proposal sources:

- **Movement/pathfinding** uses shortest discovered routes to movement frontiers
  and verified level-advance edges, then expands untried movement effects.
- **Toggle/configuration** runs a bounded beam over discovered states, scores
  visual changes, advances and remaining frontiers, and suppresses duplicate
  states.
- **Click construction** searches dynamic tile/object anchors and adds generic
  horizontal, vertical and rotational symmetry proposals around productive
  clicks.
- **Selection/commit** tests ACTION7 when it is legal and promotes it after a
  verified level transition.
- **Unknown mechanics** retains the normal DRLQN Q/prior expansion path.

A lightweight mechanic classifier measures which action families reliably
change the frame, create cycles, or advance a level. Its soft belief weights the
specialist proposals. The production graph remains authoritative and a verified
winning edge always preempts the portfolio.

Use the portfolio in the official matrix with:

```bash
PYTHONPATH=. python scripts/benchmark_drlqn_matrix.py \
  --duck-checkpoint checkpoints/duck_sft_drlqn_xs.npz \
  --portfolio off --portfolio on \
  --resume
```

The default matrix includes portfolio off/on together with budgets 32, 100, 300
and 600; seeds 0-4; cold/Duck-SFT; history off/on; graph planning off/on; and
differential complexity off/on.

## Generate offline experience from all 25 environments

The generator produces dense trajectories from random exploration, graph
frontier search, and the specialist portfolio. It retains negative evidence:
no-change actions, repeated visual states, and terminal failures.

```bash
PYTHONPATH=. python scripts/generate_offline_experience.py \
  --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 \
  --policy random --policy frontier --policy specialist \
  --max-actions-per-level 300 \
  --output data/environment_generated_transitions.npz \
  --manifest results/environment_generated_manifest.json
```

With no `--game` filter, every checked-in official environment is included. The
output uses the dense Duck NPZ format, so Duck and generated transitions can be
mixed without a second data loader.

## Four-stage training curriculum

1. **Duck SFT** behavior-clones high-quality demonstration actions.
2. **Environment world-model training** learns frame-change, level-advance and
   next-state representations from generated transitions while replaying a
   smaller Duck stream to avoid forgetting.
3. **Planner/value distillation** mixes Duck and planner trajectories. Random
   actions provide effect/value evidence but are masked from action imitation.
4. **Online adaptation** remains active during the official evaluation run.

```bash
PYTHONPATH=. python scripts/train_curriculum.py \
  --duck data/duck_dense_transitions.npz \
  --generated data/environment_generated_transitions.npz \
  --output-checkpoint checkpoints/duck_env_planner_drlqn_xs.npz \
  --metrics results/curriculum_metrics.json
```

The checkpoint is validated with the existing ordered-parameter loader. Complete
games and complete `(game, seed)` groups stay together during evaluation.

## Accelerator-backed student comparison

The optional PyTorch path compares:

- a larger action-embedding DRLQN with an action-conditioned residual executor;
- a recurrent visual policy over recent action/effect summaries;
- a compact transformer over object summaries, 8x8 tile patches and effect
  tokens;
- the larger DRLQN router without the residual executor/JVP runtime flag.

Train on an accelerator:

```bash
PYTHONPATH=. python scripts/train_accelerated_students.py \
  data/duck_dense_transitions.npz \
  data/environment_generated_transitions.npz \
  --epochs 12
```

Offline action metrics are not the final selection criterion. Run the exported
checkpoints through the official local environments:

```bash
PYTHONPATH=. python scripts/benchmark_accelerated_students.py \
  --checkpoint checkpoints/accelerated/large-drlqn.pt \
  --checkpoint checkpoints/accelerated/recurrent-visual.pt \
  --checkpoint checkpoints/accelerated/tile-transformer.pt \
  --checkpoint checkpoints/accelerated/drlqn-no-jvp.pt \
  --budget 300
```

Prefer the simplest model. Retain the sparse executor/JVP path only when it
improves held-out level completion or action efficiency, not merely imitation
accuracy.

## Submission hardening

Packaging is deliberately gated on a metrics file containing a positive
held-out level-completion gain. The bundle includes the checkpoint, checkpoint
metadata, source runtime, competition notebook, gain evidence, and an offline
wheelhouse.

```bash
PYTHONPATH=. python scripts/package_submission.py \
  --checkpoint checkpoints/duck_env_planner_drlqn_xs.npz \
  --metrics results/held_out_gain.json \
  --wheelhouse wheelhouse \
  --notebook notebooks/simplified-submission-approach.ipynb

PYTHONPATH=. python scripts/validate_submission_bundle.py \
  dist/drlqn_submission.zip --install-wheels
```

The hardened scorecard runner enforces deterministic seeds, strict per-level
action budgets, checkpoint validation, per-game traces, and telemetry:

```bash
PYTHONPATH=. python scripts/run_submission_scorecard.py \
  --checkpoint checkpoints/duck_env_planner_drlqn_xs.npz \
  --passes 1 \
  --max-actions-per-level 600
```

Multiple complementary passes require the explicit
`--allow-multiple-passes` switch and should be used only after confirming that
the active competition rules permit them.

## Hosted-data execution

The hosted Duck replay still requires `ARC_DATASET_API_KEY` at runtime. Do not
place the key in source control or workflow inputs. Supply it as a local secret
or GitHub Actions repository secret when running the replay/curriculum workflow.
