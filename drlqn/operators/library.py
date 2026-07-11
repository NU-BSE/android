"""The lambda library Lambda = {lambda_1..lambda_K}, K = 128.

  lambda_0..lambda_4         : key operators ACTION1..ACTION5
  lambda_5..lambda_124       : CLICK(slot, offset) - 24 spatial anchors x 5
                                offsets. Anchors are state-conditioned proposals
                                (objects, logical tiles, changed regions, and
                                coarse lattice points), rather than objects only.
  lambda_125..lambda_127     : extra key operators ACTION7..ACTION9

The fixed output geometry is retained for checkpoint/replay compatibility.  The
last three logits used to be permanently masked; exposing them as generic key
operators closes an absolute compatibility gap in official games that advertise
ACTION7 while preserving every existing tensor shape.
"""
from __future__ import annotations

import numpy as np

from ..env.core import ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, CLICK
from ..planning.spatial import ClickCandidate, candidate_token, generate_click_candidates
from ..state import Obj, tokenize, TOKEN_DIM
from .base import OperatorMeta

N_KEYS = 5
N_SLOTS = 24
N_OFFSETS = 5           # center, up, down, left, right
K = 128
ARG_DIM = TOKEN_DIM + N_OFFSETS

CLICK_FIRST = N_KEYS
CLICK_COUNT = N_SLOTS * N_OFFSETS
CLICK_LAST = CLICK_FIRST + CLICK_COUNT - 1
EXTRA_FIRST = CLICK_LAST + 1
EXTRA_ACTIONS = (7, 8, 9)

_KEY_ACTIONS = [ACTION1, ACTION2, ACTION3, ACTION4, ACTION5]
_OFF = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

METAS: list[OperatorMeta] = []
for i in range(N_KEYS):
    METAS.append(OperatorMeta(i, arity=0, branch_count=2,
                              kind="move" if i < 4 else "key",
                              mutates_relations=True))
for lam in range(CLICK_FIRST, CLICK_LAST + 1):
    METAS.append(OperatorMeta(lam, arity=1, branch_count=2, kind="click",
                              mutates_relations=True))
for lam in range(EXTRA_FIRST, K):
    METAS.append(OperatorMeta(lam, arity=0, branch_count=2, kind="key",
                              mutates_relations=True))

META_FEATURES = np.stack([m.features() for m in METAS])


def is_click_lambda(lam: int) -> bool:
    return CLICK_FIRST <= int(lam) <= CLICK_LAST


def is_extra_lambda(lam: int) -> bool:
    return EXTRA_FIRST <= int(lam) < K


def click_lambda(slot: int, offset: int) -> int:
    if not 0 <= int(slot) < N_SLOTS:
        raise ValueError(f"click slot must be in [0, {N_SLOTS}), got {slot}")
    if not 0 <= int(offset) < N_OFFSETS:
        raise ValueError(f"click offset must be in [0, {N_OFFSETS}), got {offset}")
    return CLICK_FIRST + int(slot) * N_OFFSETS + int(offset)


def click_slot_offset(lam: int) -> tuple[int, int]:
    if not is_click_lambda(lam):
        raise ValueError(f"lambda {lam} is not a click operator")
    r = int(lam) - CLICK_FIRST
    return r // N_OFFSETS, r % N_OFFSETS


def lambda_for_action(action: int) -> int:
    """Return the exact non-coordinate lambda for an engine action id."""
    action = int(action)
    if 1 <= action <= N_KEYS:
        return action - 1
    if action in EXTRA_ACTIONS:
        return EXTRA_FIRST + EXTRA_ACTIONS.index(action)
    raise ValueError(f"ACTION{action} has no exact key lambda")


def action_for_lambda(lam: int) -> int:
    lam = int(lam)
    if 0 <= lam < N_KEYS:
        return int(_KEY_ACTIONS[lam])
    if is_extra_lambda(lam):
        return int(EXTRA_ACTIONS[lam - EXTRA_FIRST])
    if is_click_lambda(lam):
        return CLICK
    raise ValueError(f"invalid lambda id {lam}")


def spatial_anchors(grid: np.ndarray, objs: list[Obj],
                    previous_grid: np.ndarray | None = None) -> list[ClickCandidate]:
    return generate_click_candidates(grid, objs, N_SLOTS, previous_grid)


def _action_ids(env) -> set[int]:
    values = getattr(env, "action_space", ()) or ()
    result: set[int] = set()
    for value in values:
        raw = getattr(value, "value", value)
        try:
            result.add(int(raw))
        except (TypeError, ValueError):
            continue
    return result


def valid_mask(env, anchors) -> np.ndarray:
    mask = np.zeros(K, dtype=bool)
    action_ids = _action_ids(env)
    for i, action in enumerate(_KEY_ACTIONS):
        mask[i] = int(action) in action_ids
    if bool(getattr(env, "uses_click", False)) or CLICK in action_ids:
        n = min(len(anchors), N_SLOTS)
        for slot in range(n):
            for offset in range(N_OFFSETS):
                mask[click_lambda(slot, offset)] = True
    for index, action in enumerate(EXTRA_ACTIONS):
        mask[EXTRA_FIRST + index] = action in action_ids
    return mask


def _anchor_token(anchor, grid: np.ndarray | None = None) -> np.ndarray:
    if isinstance(anchor, ClickCandidate):
        if grid is None:
            raise ValueError("grid is required to encode ClickCandidate anchors")
        return candidate_token(anchor, grid)
    # Backward-compatible object-only path for old tests/checkpoints.
    return tokenize([anchor])[0]


def arg_features(anchors, grid: np.ndarray | None = None) -> np.ndarray:
    """(K, ARG_DIM) spatial-argument features per lambda."""
    features = np.zeros((K, ARG_DIM))
    n = min(len(anchors), N_SLOTS)
    for slot in range(n):
        feat = _anchor_token(anchors[slot], grid)
        for offset in range(N_OFFSETS):
            lam = click_lambda(slot, offset)
            features[lam, :TOKEN_DIM] = feat
            features[lam, TOKEN_DIM + offset] = 1.0
    return features


def _anchor_xy_scale(anchor) -> tuple[float, float, int, int]:
    if isinstance(anchor, ClickCandidate):
        scale = max(1, int(anchor.scale))
        return float(anchor.x), float(anchor.y), scale, scale
    h = max(3, int(anchor.y1 - anchor.y0 + 1))
    w = max(3, int(anchor.x1 - anchor.x0 + 1))
    return float(anchor.cx), float(anchor.cy), w, h


def to_env_action(lam: int, anchors):
    """Map a lambda to ``(action_id, x, y)`` for local/Arcade adapters."""
    lam = int(lam)
    if not is_click_lambda(lam):
        return action_for_lambda(lam), None, None
    slot, offset = click_slot_offset(lam)
    if slot >= len(anchors):
        raise IndexError(f"lambda {lam} references missing spatial anchor {slot}")
    anchor = anchors[slot]
    cx, cy, w, h = _anchor_xy_scale(anchor)
    dy, dx = _OFF[offset]
    x = int(np.clip(round(cx + dx * w), 0, 63))
    y = int(np.clip(round(cy + dy * h), 0, 63))
    return CLICK, x, y
