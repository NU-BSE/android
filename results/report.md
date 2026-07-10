# DRLQN on ARC-AGI-3 — Final Status

Full DRLQN stack implemented per spec (pure NumPy — PyPI/torch blocked in sandbox):
64x64 columns (E=4096) / 64x16 control (I=1024) geometry (DRLQN-M, 47.6M params,
instantiates & passes hard-path checks; DRLQN-S 11.3M), K=128 lambda-actions with
object-typed 256-d embeddings, Double+Dueling Q-router, hard Top-k=6 column routing,
sparse residual lambda-executor (per-lambda low-rank adapters + FiLM), hand-written
forward-mode JVP differential shadow with branch/Top-k/remainder guards, exact
fallback, audits & re-anchoring. Training runs DRLQN-XS on CPU (test-time RL,
first-contact runs). All P0 tests pass (gradcheck, JVP vs finite differences,
env determinism/solvability/random-play resistance).

## Solve status (5 local ARC-AGI-3-style envs x 6 levels, 12x-baseline budget)
- xg01 (Passage):  4/6 levels — L5 (teleporters) and L6 not reached within budget.
- rt02 (Rotor):    1-4/6 across runs — later linked-tile levels NOT solved.
- fl03 (Conduit):  1/6 — L2+ NOT solved (combinatorial toggle space defeats
  exploration within budget).
- pb04 (Pusher):   1/6 — L2+ NOT solved (sparse reward, no goal signal pre-win).
- sq05 (Chroma):   2-3/6 — hidden-cue / legend levels NOT solved within budget.

RHAE (best full run): ~0.08-0.10. NOT TRAINED TO COMPLETION: the un-solved
levels above were not further approached (per instruction); they would need
longer test-time-training budgets or model-based planning over the learned
executor. Real ARC-AGI-3 API untested (no ARC_API_KEY in sandbox; adapter in
drlqn/api/, scripts/run_api.py). Frontier-model reference on the real
benchmark is <1%, for context.

See results/results.json + results/solve_all.log for the final run's numbers
(run may have been cut by session termination; partial rows are valid).
