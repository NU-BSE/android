# DRLQN on ARC-AGI-3

DRLQN (Differential Residual λ Q-Network) agent for ARC-AGI-3-style
interactive environments, implementing the DRLQN-M design note:

- **64 logical columns × 64 dims (E=4096), 64 control slots × 16 dims
  (I=1024)** — interneuron-inspired sparse control prior (E:I ≈ 4:1)
- **K=128 λ-actions with 256-d embeddings** — 5 key operators + object-typed
  click operators (24 object slots × 5 offsets); action embeddings combine a
  learned library embedding, fixed operator metadata, and the token features
  of the object argument
- **Double + Dueling DQN Q-router** with invalid-action masking
- **Hard Top-k column routing (k=6)** — authoritative sparse mask
- **Sparse residual λ-executor** — B residual blocks with per-λ low-rank
  adapters + FiLM, run only on active columns; λ : S → ΔS
- **Differential shadow** — hand-written forward-mode JVP (exact equivalent
  of `torch.func.jvp`; PyPI is unreachable in the build sandbox, so the whole
  stack runs on a ~300-line numpy autodiff engine, `drlqn/autodiff.py`) with
  branch-margin / Top-k / remainder stability guards, exact fallback,
  periodic audits and re-anchoring. Advisory only — hard semantics decide.

## Layout (spec §8.1)

```
drlqn/
  autodiff.py            numpy reverse-mode engine (torch unavailable)
  config.py              DRLQN-XS / S / M geometries
  state.py               canonical Markov state, objects, semantic deltas
  env/core.py            ARC-AGI-3 environment format (64x64, 16 colors)
  env/games/             5 local environments x 6 levels (xg01 rt02 fl03 pb04 sq05)
  operators/             λ library, metadata, RADDT-style margins/shadow
  model/                 relational encoder, controller, Q-router,
                         column router, sparse residual λ-executor
  runtime/               NodeCache, JVP, stability guards, approximate
                         recompute with exact fallback, shadow
  train/                 replay, losses (L_Q, L_ΔQ, L_op, L_router), solver
  eval/                  scripted baselines + RHAE scoring
  api/                   three.arcprize.org adapter (needs ARC_API_KEY)
scripts/solve_all.py     solve + score everything -> results/results.json
scripts/run_api.py       same agent against the official API
tests/                   P0 verification: gradcheck, JVP vs finite
                         differences, env determinism/solvability, scales
```

## Run

```
PYTHONPATH=. python tests/test_autodiff.py
PYTHONPATH=. python tests/test_jvp.py
PYTHONPATH=. python tests/test_env.py
PYTHONPATH=. python tests/test_scale.py
python scripts/solve_all.py
```

## Notes / deviations

- The real ARC-AGI-3 environments live behind `three.arcprize.org` and need
  an ARC API key that this sandbox does not have; the five local
  environments follow the published format and design constraints
  (levels, tutorial first level, composed mechanics, no language symbols).
- Training runs the DRLQN-XS geometry (CPU-only numpy); DRLQN-S (11.3M) and
  DRLQN-M (47.6M) instantiate and pass hard-path forward checks
  (`tests/test_scale.py`) with identical semantics.
- Human baselines are surrogate (`ceil(1.75·optimal)+5`); no human cohort.
- See `results/report.md` for solve results, RHAE scores and runtime
  telemetry (JVP acceptance, guard failures, drift audits).
