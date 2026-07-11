# DRLQN top-performance route

## What the accessible implementations change

The checked-in competition package contains 25 ARCEngine games rather than the
five synthetic DRLQN environments used in the original report. Static audit of
the declared action spaces finds:

- 17 games expose coordinate action `ACTION6`;
- six games expose `ACTION7`: `ar25`, `bp35`, `lf52`, `sb26`, `sk48`, and `su15`;
- no checked-in game requires an action above `ACTION7`.

The last three DRLQN logits were previously reserved and permanently masked, so
six games had an absolute interface failure before any learning or planning
quality was considered. Lambdas 125--127 now map to `ACTION7`--`ACTION9` while
keeping `K=128`, replay shapes, model shapes, and older checkpoints compatible.

ARCEngine's reset implementation also confirms that a reset after at least one
action resets the current level, while reset at action count zero or after a win
performs a full reset. The graph solver's use of level reset is therefore valid
for local competition environments.

## Why Duck Harness performs differently

Duck's main advantage is not a more elaborate value-network block. Its agent has
several forms of information DRLQN did not have:

1. a pretrained language-model prior;
2. a structured scene description and explicit valid-action list;
3. a persistent transcript and world-model notes;
4. arbitrary Python search over a hypothesized transition system;
5. exact coordinate actions and batched plans;
6. repeated passes with retained findings;
7. trace export, scoring, and run instrumentation.

The original DRLQN starts from random `DRLQN-XS` weights and learns only from
sparse first-contact experience. Its trace vector was constant, its state key was
visual-only, its click coordinates were object-centroid offsets, and its graph
planner ranked little beyond untried edges. Differential/JVP machinery cannot
compensate for those missing priors and interfaces.

## Implemented route

### 1. Preserve the fixed action geometry, remove compatibility failures

- `ACTION7`--`ACTION9` use the formerly reserved logits.
- Dynamic `response.available_actions` masks invalid keys at every observation.
- `ACTION6` remains 24 state-conditioned anchors times five offsets.
- Semantic objects remain encoder tokens; click anchors are a separate action
  representation. This corrects the first hybrid version, which weakened object
  reasoning by replacing semantic objects with click proposals.

### 2. Add bounded interaction memory and stronger graph routing

`SubmissionDRLQNSolver` keys a node by grid, level, and a short bounded sequence
of `(lambda, frame_changed, level_advanced)` effects. Absolute step count remains
a neural trace feature but is deliberately excluded from graph identity, so the
same contextual state can be revisited.

The solver records action effect rates and successful level-transition edges,
replays known winning edges first, and routes toward value-ranked frontiers
before resetting. Movement, salience, changed-region, logical-tile, and click
proposal priors remain low-cost exploration biases rather than hard-coded game
solutions.

### 3. Add the official runtime boundary

`ArcadeEnvAdapter` translates `arc_agi.Arcade` responses to the existing solver
API and treats each response's `available_actions` as authoritative. The runner
in `scripts/solve_arcade.py` works with local competition environments or a
Kaggle rerun and can load a distilled NumPy checkpoint.

### 4. Distill Duck traces before online RL

Duck's trace export attaches a 64x64 grid to user messages and Python tool calls
to assistant messages. A Python call can batch several actions, but intermediate
frames are not guaranteed to be present. The loader therefore keeps only the
first action after each fresh grid as a trustworthy state/action pair.

`DuckSFTTrainer` behavior-clones the DRLQN encoder and Q router. Key actions map
exactly; click actions map to the nearest current spatial proposal and are
weighted by coordinate distance. The resulting ordered NumPy parameter list is
saved with shape and count validation and can initialize each Arcade game before
online adaptation.

Example workflow:

```bash
python scripts/distill_duck.py /path/to/exported/duck/traces \
  --epochs 8 \
  --batch-size 16 \
  --output-checkpoint checkpoints/duck_sft_drlqn_xs.npz

python scripts/audit_environment_files.py \
  arc-prize-2026-arc-agi-3/environment_files

python scripts/solve_arcade.py \
  --checkpoint checkpoints/duck_sft_drlqn_xs.npz \
  --max-actions-per-level 600
```

## Validation completed

CI now covers the legacy autodiff/JVP/local-environment tests, spatial proposal
coverage, bounded history, extra-action mapping, Duck trace parsing, one SFT
optimization step, checkpoint round-trip, and the Arcade adapter. A separate job
installs `arcengine==0.9.3`, imports all 25 checked-in games, performs RESET, and
executes one advertised legal action through the submission adapter in every
game.

This is runtime and interface validation, not a competition score. No claim of
leaderboard performance should be made until a distilled checkpoint is trained
on real Duck traces and the full 25-game Arcade run is scored.

## Highest-value next experiments

1. **Generate teacher coverage.** Run Duck on all available games, seeds, and
   repeated passes; retain successful and informative failed transitions.
2. **Add auxiliary distillation targets.** Predict frame change, level advance,
   next-state embedding, and teacher value/rank in addition to the chosen action.
3. **Use mechanic specialists.** Dispatch among movement/pathfinding,
   toggle/search, click-construction, and selection/`ACTION7` planners. The
   shared graph should supply exact transitions to BFS/beam/MCTS rather than ask
   a random Q network to rediscover planning from scratch.
4. **Train offline before test-time adaptation.** Mix teacher traces with
   trajectories generated directly from the provided ARCEngine games. Split by
   game/seed/level family to detect memorization.
5. **Scale the student.** Move the student to an accelerator-backed framework or
   distill into a compact vision-policy model. `DRLQN-XS` is useful for proving
   the interface but is unlikely to be the final-capacity model.
6. **Ablate complexity.** Compare the sparse residual executor and JVP shadow
   against a simpler recurrent/transformer policy once pretraining and planning
   are present. Retain them only if they improve held-out score or action
   efficiency.
7. **Use a portfolio.** Run complementary policy/planner seeds or passes and
   preserve solved mechanics and successful routes. Duck's repeated-pass design
   is a more credible route to robust top performance than a single cold online
   RL trajectory.
