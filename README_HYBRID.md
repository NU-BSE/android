# Hybrid spatial DRLQN quickstart

Run the original tests plus the new spatial/distillation regressions:

```bash
PYTHONPATH=. pytest -q tests
```

Run the opt-in hybrid solver on the five local environments:

```bash
PYTHONPATH=. python scripts/solve_hybrid.py
```

Audit a mounted ARC environment package tree before submission:

```bash
python scripts/audit_environment_files.py \
  /arc-prize-2026-arc-agi-3/environment_files \
  --json results/environment_audit.json
```

The hybrid solver intentionally does not replace `DRLQNSolver`; use
`drlqn.train.hybrid_solver.HybridDRLQNSolver` for side-by-side ablations.
