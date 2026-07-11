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


_FORMAT = "drlqn-ordered-parameters-v1"


def save_model_checkpoint(model, path: str | Path,
                          metadata: dict[str, Any] | None = None) -> Path:
    """Save the ordered parameter list returned by the numpy ``Module`` API."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = [np.asarray(value) for value in model.state_dict()]
    payload = {
        f"param_{index:05d}": value
        for index, value in enumerate(state)
    }
    np.savez_compressed(path, **payload)

    sidecar = dict(metadata or {})
    sidecar.setdefault("format", _FORMAT)
    sidecar.setdefault("parameter_count", len(state))
    sidecar.setdefault("parameter_shapes", [list(value.shape) for value in state])
    meta_path = path.with_suffix(path.suffix + ".json")
    meta_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def load_model_checkpoint(model, path: str | Path) -> dict[str, Any]:
    """Load a checkpoint after validating parameter count and every shape."""
    path = Path(path)
    expected = model.state_dict()
    with np.load(path, allow_pickle=False) as archive:
        names = sorted(name for name in archive.files if name.startswith("param_"))
        state = [archive[name].copy() for name in names]

    if len(state) != len(expected):
        raise ValueError(
            f"checkpoint parameter count {len(state)} does not match model {len(expected)}"
        )
    for index, (actual, target) in enumerate(zip(state, expected)):
        if actual.shape != target.shape:
            raise ValueError(
                f"checkpoint parameter {index} shape {actual.shape} "
                f"does not match model {target.shape}"
            )
    model.load_state_dict(state)

    meta_path = path.with_suffix(path.suffix + ".json")
    if not meta_path.exists():
        return {}
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    format_name = metadata.get("format")
    if format_name not in (None, _FORMAT):
        raise ValueError(f"unsupported checkpoint format: {format_name}")
    return metadata


def load_solver_checkpoint(solver, path: str | Path) -> dict[str, Any]:
    metadata = load_model_checkpoint(solver.model, path)
    solver.target.load_state_dict(solver.model.state_dict())
    solver.caches.clear()
    return metadata
