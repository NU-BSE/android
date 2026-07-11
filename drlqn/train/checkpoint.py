"""Portable DRLQN model checkpoint helpers.

Only model parameters are persisted. Replay, graph state, and optimizer moments
remain episode-local, which keeps distilled checkpoints small and safe to load
inside a Kaggle submission notebook.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_model_checkpoint(model, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: np.asarray(value) for name, value in model.state_dict().items()}
    np.savez_compressed(path, **state)
    if metadata is not None:
        meta_path = path.with_suffix(path.suffix + ".json")
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_model_checkpoint(model, path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        state = {name: archive[name].copy() for name in archive.files}
    model.load_state_dict(state)
    meta_path = path.with_suffix(path.suffix + ".json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def load_solver_checkpoint(solver, path: str | Path) -> dict[str, Any]:
    metadata = load_model_checkpoint(solver.model, path)
    solver.target.load_state_dict(solver.model.state_dict())
    solver.caches.clear()
    return metadata
