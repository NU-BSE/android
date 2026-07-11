"""The lambda library Lambda = {lambda_1..lambda_K}, K = 128.

  lambda_0..lambda_4         : key operators ACTION1..ACTION5
  lambda_5..lambda_124       : CLICK(slot, offset) - 24 spatial anchors x 5
                                offsets. Anchors are state-conditioned proposals
                                (objects, logical tiles, changed regions, and
                                coarse lattice points), rather than objects only.
  lambda_125..lambda_127     : reserved (always masked invalid)

The fixed output geometry is retained for checkpoint/replay compatibility.  The
proposal generator broadens ACTION6 coverage to empty cells and merged logical
cells, closing the largest representational gap relative to tool-using agents.
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

_KEY_ACTIONS = [ACTION1, ACTION2, ACTION3, ACTION4, ACTION5]
_OFF = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

METAS: list[OperatorMeta] = []
for i in range(N_KEYS):
    METAS.append(OperatorMeta(i, arity=0, branch_count=2,
                              kind="move" if i < 4 else "key",
                              mutates_relations=True))
for lam in range(N_KEYS, K):
    METAS.append(OperatorMeta(lam, arity=1, branch_count=2, kind="click",
                              mutates_relations=True))

META_FEATURES = np.stack([m.features() for m in METAS])


def click_lambda(slot: int, offset: int) -> int:
    return N_KEYS + slot * N_OFFSETS + offset


def click_slot_offset(lam: int) -> tuple[int, int]:
    r = lam - N_KEYS
    return r // N_OFFSETS, r % N_OFFSETS


def spatial_anchors(grid: np.ndarray, objs: list[Obj],
                    previous_grid: np.ndarray | None = None) -> list[ClickCandidate]:
    return generate_click_candidates(grid, objs, N_SLOTS, previous_grid)


def valid_mask(env, anchors) -> np.ndarray:
    mask = np.zeros(K, dtype=bool)
    for i, a in enumerate(_KEY_ACTIONS):
        mask[i] = a in env.action_space
    if env.uses_click:
        n = min(len(anchors), N_SLOTS)
        for s in range(n):
            for o in range(N_OFFSETS):
                mask[click_lambda(s, o)] = True
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
    F = np.zeros((K, ARG_DIM))
    n = min(len(anchors), N_SLOTS)
    for s in range(n):
        feat = _anchor_token(anchors[s], grid)
        for o in range(N_OFFSETS):
            lam = click_lambda(s, o)
            F[lam, :TOKEN_DIM] = feat
            F[lam, TOKEN_DIM + o] = 1.0
    return F


def _anchor_xy_scale(anchor) -> tuple[float, float, int, int]:
    if isinstance(anchor, ClickCandidate):
        scale = max(1, int(anchor.scale))
        return float(anchor.x), float(anchor.y), scale, scale
    h = max(3, int(anchor.y1 - anchor.y0 + 1))
    w = max(3, int(anchor.x1 - anchor.x0 + 1))
    return float(anchor.cx), float(anchor.cy), w, h


def to_env_action(lam: int, anchors):
    """Map a lambda to (action, x, y) for ArcEnv.step / remote API."""
    if lam < N_KEYS:
        return _KEY_ACTIONS[lam], None, None
    slot, off = click_slot_offset(lam)
    anchor = anchors[slot]
    cx, cy, w, h = _anchor_xy_scale(anchor)
    dy, dx = _OFF[off]
    x = int(np.clip(round(cx + dx * w), 0, 63))
    y = int(np.clip(round(cy + dy * h), 0, 63))
    return CLICK, x, y
