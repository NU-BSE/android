from __future__ import annotations

import argparse
import importlib
import os
from collections.abc import Callable
from typing import Any

import optuna

from clients.python_client import ArcDatasetClient

Objective = Callable[[optuna.Trial, ArcDatasetClient], float]


def load_objective(import_path: str) -> Objective:
    """Load `package.module:function` supplied by the training project."""
    if ":" not in import_path:
        raise ValueError("Objective must use package.module:function syntax")
    module_name, function_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    objective = getattr(module, function_name)
    if not callable(objective):
        raise TypeError(f"{import_path} is not callable")
    return objective


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an Optuna study whose objective consumes the hosted ARC dataset API."
    )
    parser.add_argument("--objective", required=True, help="package.module:function")
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--direction", choices=["minimize", "maximize"], default="maximize")
    parser.add_argument(
        "--storage",
        default=os.getenv("OPTUNA_STORAGE_URL"),
        help="Optuna RDB URL, normally a separate PostgreSQL database/schema",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.getenv("ARC_DATASET_API_URL", "http://localhost:8000"),
    )
    parser.add_argument("--api-key", default=os.getenv("ARC_DATASET_API_KEY"))
    args = parser.parse_args()

    if not args.storage:
        raise SystemExit("Set --storage or OPTUNA_STORAGE_URL")

    user_objective = load_objective(args.objective)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=args.direction,
        load_if_exists=True,
    )

    with ArcDatasetClient(args.api_base_url, args.api_key) as dataset:
        study.optimize(lambda trial: user_objective(trial, dataset), n_trials=args.trials)

    print(f"Best value: {study.best_value}")
    print(f"Best parameters: {study.best_params}")


if __name__ == "__main__":
    main()
