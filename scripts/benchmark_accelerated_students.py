"""Run accelerator student checkpoints on the checked-in official environments.

This is the completion/action-efficiency gate that complements offline held-out
classification metrics. The model chooses lambda actions through the same
rendered-state perception and dynamic valid mask used by the numpy solver.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from drlqn.accelerator.models import build_accelerator_model, require_torch
from drlqn.config import drlqn_xs
from drlqn.env.official_loader import environment_files, load_arcengine, make_arcade_env
from drlqn.operators.library import to_env_action
from drlqn.train.submission_solver import SubmissionDRLQNSolver


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("arc-prize-2026-arc-agi-3/environment_files"),
    )
    parser.add_argument("--game", action="append", default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--max-levels", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/accelerated_official_benchmark.json"),
    )
    return parser.parse_args()


def _effect(
    action: int,
    changed: bool,
    advanced: bool,
    level: int,
    x,
    y,
) -> np.ndarray:
    return np.asarray(
        [
            float(action) / 9.0,
            float(changed),
            float(advanced),
            float(level) / 128.0,
            float(-1 if x is None else x) / 64.0,
            float(-1 if y is None else y) / 64.0,
        ],
        dtype=np.float32,
    )


def _load_model(torch, checkpoint: Path, device):
    payload = torch.load(checkpoint, map_location=device)
    config = dict(payload.get("config") or {})
    name = str(config.get("name") or checkpoint.stem)
    overrides = {key: value for key, value in config.items() if key != "name"}
    model = build_accelerator_model(name, **overrides).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, config


def _solve(
    torch,
    model,
    path: Path,
    arcengine: Any,
    *,
    seed: int,
    budget: int,
    max_levels: int,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_arcade_env(path, arcengine, seed=seed)
    perceiver = SubmissionDRLQNSolver(drlqn_xs(), seed=seed, history_len=4)
    frame = env.reset()
    node = perceiver.perceive(env, frame.grid)
    previous_effect = np.zeros(6, dtype=np.float32)
    hidden = None
    no_change = 0
    advances: list[int] = []

    while env.state == "NOT_FINISHED":
        level = int(env.level)
        if level > max_levels:
            break
        used = int(env._level_action_counts.get(level, 0))
        if used >= budget:
            break
        valid = np.asarray(node.valid, dtype=bool)
        if not np.any(valid):
            break

        grid = torch.as_tensor(
            np.asarray(node.grid, dtype=np.int64)[None],
            device=next(model.parameters()).device,
            dtype=torch.long,
        )
        effect = torch.as_tensor(
            previous_effect[None],
            device=grid.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            output = model(grid, effect, hidden=hidden)
        logits = output["logits"][0].detach().cpu().numpy()
        logits = np.where(valid, logits, -1e30)
        lam = int(np.argmax(logits))
        perceiver._pending_lambda = lam
        action, x, y = to_env_action(lam, node.objs)
        before = np.asarray(node.grid, dtype=np.uint8).copy()
        level_before = int(env.level)
        step = env.step(action, x, y)
        nxt = perceiver.perceive(env, step.frame.grid)
        changed = not np.array_equal(before, nxt.grid)
        advanced = bool(step.level_won or int(env.level) > level_before)
        no_change += int(not changed)
        if advanced:
            advances.append(int(env.action_count))

        perceiver.graph.setdefault(node.key, {})[lam] = nxt.key
        perceiver._update_priors(node, lam, nxt, changed)
        previous_effect = _effect(action, changed, advanced, level_before, x, y)
        hidden = output.get("hidden")
        if hidden is not None:
            hidden = hidden.detach()
        node = nxt
        if advanced:
            previous_effect = np.zeros(6, dtype=np.float32)
            hidden = None
        if step.env_won or env.state == "GAME_OVER":
            break

    levels = int(getattr(env, "_levels_completed", 0))
    actions = int(env.action_count)
    return {
        "game": path.stem,
        "seed": int(seed),
        "won": bool(env.state == "WIN"),
        "levels_completed": levels,
        "total_actions": actions,
        "actions_per_level": env.actions_per_level,
        "first_level_advance_action": advances[0] if advances else None,
        "level_advance_actions": advances,
        "no_change_actions": int(no_change),
        "action_efficiency": float(levels / max(actions, 1)),
    }


def _aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        if not row.get("error"):
            groups[row["model"]].append(row)
    result = []
    for model, group in sorted(groups.items()):
        advances = [
            row["first_level_advance_action"]
            for row in group
            if row["first_level_advance_action"] is not None
        ]
        result.append(
            {
                "model": model,
                "runs": len(group),
                "games": len({row["game"] for row in group}),
                "seeds": len({row["seed"] for row in group}),
                "win_rate": float(np.mean([row["won"] for row in group])),
                "mean_levels_completed": float(
                    np.mean([row["levels_completed"] for row in group])
                ),
                "mean_total_actions": float(
                    np.mean([row["total_actions"] for row in group])
                ),
                "mean_action_efficiency": float(
                    np.mean([row["action_efficiency"] for row in group])
                ),
                "mean_no_change_rate": float(
                    np.mean(
                        [
                            row["no_change_actions"] / max(row["total_actions"], 1)
                            for row in group
                        ]
                    )
                ),
                "median_first_level_advance_action": (
                    float(np.median(advances)) if advances else None
                ),
            }
        )
    return result


def main() -> int:
    args = _parse_args()
    torch, _nn = require_torch()
    if args.budget <= 0 or args.max_levels <= 0:
        raise SystemExit("budget and max levels must be positive")
    for checkpoint in args.checkpoint:
        if not checkpoint.exists():
            raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    torch.use_deterministic_algorithms(True, warn_only=True)
    mps = getattr(torch.backends, "mps", None)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if mps and mps.is_available()
        else "cpu"
    )
    arcengine = load_arcengine()
    files = environment_files(args.root)
    selected = set(args.game)
    if selected:
        files = [path for path in files if path.stem in selected]
    if not files:
        raise SystemExit(f"no environments found under {args.root}")
    seeds = tuple(args.seed or range(5))

    rows = []
    errors = 0
    for checkpoint in args.checkpoint:
        model, config = _load_model(torch, checkpoint, device)
        name = str(config.get("name") or checkpoint.stem)
        for path in files:
            for seed in seeds:
                try:
                    row = _solve(
                        torch,
                        model,
                        path,
                        arcengine,
                        seed=int(seed),
                        budget=int(args.budget),
                        max_levels=int(args.max_levels),
                    )
                    row.update(
                        {
                            "model": name,
                            "checkpoint": str(checkpoint),
                            "differential_runtime": bool(
                                config.get("differential_runtime", False)
                            ),
                        }
                    )
                except Exception as exc:
                    errors += 1
                    row = {
                        "model": name,
                        "checkpoint": str(checkpoint),
                        "game": path.stem,
                        "seed": int(seed),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(row)
                print(json.dumps(row, sort_keys=True))

    payload = {
        "device": str(device),
        "budget": int(args.budget),
        "seeds": list(seeds),
        "games": [path.stem for path in files],
        "aggregate": _aggregate(rows),
        "rows": rows,
        "selection_rule": (
            "prefer the simplest student; keep the sparse executor/JVP path only "
            "when it improves held-out completion or action efficiency"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
