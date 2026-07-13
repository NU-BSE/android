# Dense Duck replay and DRLQN distillation

Duck Harness may issue several ARC actions inside one Python tool call while the
trace records only the grid before the call and a later grid. The dense replay
pipeline executes those actions against the checked-in official environment,
verifies each later observed frame, and emits every intermediate transition.

## Frontend integration

Copy `dataset-host/clients/frontend.ts` into the frontend or import it from this
repository. The client preserves reverse-proxy prefixes such as `/agi-3` and
supports API-key authentication.

```ts
import { ArcDatasetApi } from "./frontend";

const dataset = new ArcDatasetApi(
  "http://hse.onl/agi-3",
  process.env.ARC_DATASET_API_KEY,
);
const recordings = await dataset.listRecordings("aa00");
const metadata = await dataset.getRecording(recordings[0].id);
console.log(metadata.game_id);

for await (const event of dataset.streamRecording(recordings[0].id)) {
  console.log(event);
}
```

Do not commit the dataset API key. Set it through the deployment secret store or
the `ARC_DATASET_API_KEY` environment variable.

## Reconstruct dense trajectories

The replayer reads recording metadata and NDJSON events from the hosted dataset,
finds the matching local environment and process seed from the initial frame,
executes every action in each Duck batch, and checks alignment at all later
observed frames.

```bash
export ARC_DATASET_API_KEY=...                 # supplied out of band
PYTHONPATH=. python scripts/replay_duck_trajectories.py \
  --base-url http://hse.onl/agi-3 \
  --game aa00 \
  --output data/duck_dense_transitions.npz \
  --report results/duck_replay_report.json
```

Each transition stores the current and next 64×64 grids, primitive action and
click coordinate, visual-change flag, level transition, terminal state, seed,
trajectory outcome, and remaining path length. Alignment is strict by default;
`--allow-mismatch` is intended only for diagnosis and records every discrepancy.

## Distill a checkpoint

```bash
PYTHONPATH=. python scripts/distill_duck.py \
  data/duck_dense_transitions.npz \
  --epochs 12 \
  --output-checkpoint checkpoints/duck_sft_drlqn_xs.npz \
  --metrics results/duck_sft_metrics.json
```

The action router is behavior-cloned while temporary auxiliary heads predict
frame change, level advance, and the next-state embedding. Successful paths have
higher weight; visually informative failures remain in training. Evaluation
holds out complete games and complete `(game, seed)` groups. Reported metrics
include key-action accuracy, click-coordinate and proposal coverage,
level-advance recall, next-embedding cosine, and per-game held-out performance.
Only DRLQN model parameters are persisted, so the existing validated checkpoint
format remains submission-compatible.

## Cold-versus-distilled benchmark matrix

After the distilled checkpoint exists, the default command runs all requested
budgets, five seeds, and every history/planning/differential ablation:

```bash
PYTHONPATH=. python scripts/benchmark_drlqn_matrix.py \
  --duck-checkpoint checkpoints/duck_sft_drlqn_xs.npz \
  --rows results/drlqn_baseline_matrix.jsonl \
  --output results/drlqn_baseline_matrix.json \
  --resume
```

Defaults are 32, 100, 300, and 600 actions per level; seeds 0–4; cold and
Duck-SFT initialization; history off/on; graph planning off/on; and differential
JVP/executor complexity off/on. The JSONL file is append-safe for long runs.
Besides wins and completed levels, the aggregate reports whether online learning
activated and whether the first level advance happened before the replay warmup
threshold. That directly tests whether SFT reaches early rewards before online
RL begins.
