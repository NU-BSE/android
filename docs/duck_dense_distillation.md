# Dense Duck replay, planner portfolio and DRLQN curriculum

Duck Harness may issue several ARC actions inside one Python tool call while the
trace records only the grid before the call and a later grid. The dense replay
pipeline executes those actions against the checked-in official environment,
verifies every later observed frame, and emits every intermediate transition.

## Hosted recording integration

Copy `dataset-host/clients/frontend.ts` into the frontend or import it directly.
The client preserves reverse-proxy prefixes such as `/agi-3`, exposes recording
metadata, streams NDJSON, and sends `X-API-Key` when configured. Keep the key in a
secret store or `ARC_DATASET_API_KEY`; never commit it.

```ts
import { ArcDatasetApi } from "./frontend";

const dataset = new ArcDatasetApi(
  "https://hse.onl/agi-3",
  process.env.ARC_DATASET_API_KEY,
);
const recordings = await dataset.listRecordings("aa00");
const metadata = await dataset.getRecording(recordings[0].id);
for await (const event of dataset.streamRecording(metadata.id)) {
  console.log(event);
}
```

## 1. Reconstruct dense Duck trajectories

```bash
export ARC_DATASET_API_KEY=...  # supplied out of band
PYTHONPATH=. python scripts/replay_duck_trajectories.py \
  --base-url https://hse.onl/agi-3 \
  --output data/duck_dense_transitions.npz \
  --report results/duck_replay_report.json
```

The replayer resolves the local game and process seed from metadata plus the initial
frame, replays repeated and batched actions, and checks every later anchor frame.
Each transition stores current/next grids, action and click coordinate, visual
change, level advance, terminal state, seed, outcome, and remaining path length.

## 2. Generic planner portfolio

`drlqn/planning/portfolio.py` blends specialists sharing the same discovered
transition graph:

- movement/pathfinding: BFS to known wins or movement frontiers;
- toggle/configuration: beam search with state deduplication;
- click construction: tile/changed-region proposals plus rendered-space symmetry;
- selection/commit: ACTION7 hypothesis tests and verified commit reuse;
- unknown mechanics: the normal DRLQN Q/prior exploration route.

A soft mechanic classifier uses only observed changed/no-op/advance rates, action
families, graph cycles and legal actions. No game identifier or environment source
is inspected. Enable it through `planner_portfolio=True` or benchmark it with the
`--portfolio off/on` matrix axis.

## 3. Generate experience from all 25 environments

```bash
PYTHONPATH=. python scripts/generate_offline_experience.py \
  --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 \
  --policy random --policy frontier --policy specialist \
  --max-actions-per-level 300 \
  --output data/environment_generated_transitions.npz
```

The generated data labels no-change actions, cycles/regressions, terminal failures,
level advances and wins. Random actions are retained for world-model and negative
evidence training, but are not behavior-cloned as planner decisions.

## 4. Four-stage curriculum

```bash
PYTHONPATH=. python scripts/train_curriculum.py \
  --duck data/duck_dense_transitions.npz \
  --generated data/environment_generated_transitions.npz \
  --output-checkpoint checkpoints/duck_env_planner_drlqn_xs.npz \
  --metrics results/curriculum_metrics.json
```

The stages are:

1. Duck behavior cloning with frame-change, level-advance and next-state heads;
2. environment-generated transition/world-model training;
3. planner-choice and progress/no-op value distillation into persistent Q logits;
4. normal online adaptation during evaluation.

Evaluation splits whole games and whole `(game, seed)` groups. The existing ordered
numpy checkpoint format is retained and round-trip validated.

## 5. Cold, distilled and portfolio benchmark matrix

```bash
PYTHONPATH=. python scripts/benchmark_drlqn_matrix.py \
  --duck-checkpoint checkpoints/duck_env_planner_drlqn_xs.npz \
  --rows results/drlqn_baseline_matrix.jsonl \
  --output results/drlqn_baseline_matrix.json \
  --resume
```

Defaults cover budgets 32/100/300/600, seeds 0–4, cold/Duck-SFT, history off/on,
graph planning off/on, planner portfolio off/on, and JVP/executor complexity off/on.
Use explicit flags to shard the matrix across machines. The aggregate reports wins,
levels, action efficiency, online-learning activation and level advances before
warmup.

## 6. Accelerator-backed architecture ablations

```bash
python scripts/train_accelerated_students.py \
  data/duck_dense_transitions.npz \
  data/environment_generated_transitions.npz
```

The optional PyTorch runner compares a larger action-embedding DRLQN, recurrent
visual policy, tile/effect transformer, and DRLQN without differential runtime.
Retain the sparse executor/JVP path only when held-out level completion or action
efficiency improves; parameter count or architectural novelty is not a selection
criterion.

## 7. Submission hardening

Packaging is deliberately gated on a positive held-out level-completion gain:

```bash
python scripts/package_submission.py \
  --checkpoint checkpoints/duck_env_planner_drlqn_xs.npz \
  --metrics results/local_gain_report.json \
  --wheelhouse wheelhouse
python scripts/validate_submission_bundle.py \
  dist/drlqn_submission.zip --install-wheels
```

The hardened scorecard runner enforces deterministic seeds and per-level budgets,
saves per-game action/frame-hash traces and telemetry, loads the checkpoint before
every pass, and can run complementary generic configurations when competition rules
permit:

```bash
PYTHONPATH=. python scripts/run_submission_scorecard.py \
  --checkpoint checkpoints/duck_env_planner_drlqn_xs.npz \
  --passes 1 --max-actions-per-level 600
```

Keep `--passes 1` unless the current submission rules explicitly allow multiple
independent passes.
