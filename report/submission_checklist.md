# Hybrid submission checklist

1. Run `pytest -q tests`.
2. Run `scripts/audit_environment_files.py` on the mounted competition environment tree.
3. Run `scripts/solve_hybrid.py` and compare level completion, RHAE, action counts and wall time against the original solver using identical seeds/budgets.
4. Reject the branch if spatial proposal coverage improves but action efficiency or local solve count regresses materially.
5. Before leaderboard use, train on offline demonstrations or Duck traces; the current branch improves representability but does not supply pretrained policy knowledge.
6. Package the exact tested commit and record environment versions, seed portfolio, runtime and failure counts.
