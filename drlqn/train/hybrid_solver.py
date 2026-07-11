"""Hybrid DRLQN with Duck-inspired state-conditioned spatial actions.

This keeps the learned DRLQN router/executor and online training loop intact,
but replaces the object-only ACTION6 slots with a diverse proposal set covering
logical tile centres, changed regions, object boundaries and a coarse lattice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..operators.library import arg_features, spatial_anchors, valid_mask
from ..state import Obj, TOKEN_DIM, tokenize
from .trainer import DRLQNSolver, NodeInfo


@dataclass
class SpatialAnchor:
    """Obj-compatible view of a spatial proposal for legacy solver priors."""

    color: int
    y0: int
    x0: int
    y1: int
    x1: int
    cy: float
    cx: float
    area: int
    fill: float
    perim: float
    shape3: np.ndarray
    kind: str = "spatial"
    proposal_score: float = 0.0


def _as_obj(candidate, grid: np.ndarray) -> SpatialAnchor:
    s = max(1, int(candidate.scale))
    half = max(0, s // 2)
    x0, x1 = max(0, candidate.x - half), min(63, candidate.x + half)
    y0, y1 = max(0, candidate.y - half), min(63, candidate.y + half)
    color = int(grid[candidate.y, candidate.x])
    shape = np.zeros((3, 3), dtype=np.float64)
    for gy, dy in enumerate((-s, 0, s)):
        for gx, dx in enumerate((-s, 0, s)):
            yy = int(np.clip(candidate.y + dy, 0, 63))
            xx = int(np.clip(candidate.x + dx, 0, 63))
            shape[gy, gx] = float(int(grid[yy, xx]) == color)
    return SpatialAnchor(
        color=color,
        y0=y0,
        x0=x0,
        y1=y1,
        x1=x1,
        cy=float(candidate.y),
        cx=float(candidate.x),
        area=max(1, (y1 - y0 + 1) * (x1 - x0 + 1)),
        fill=1.0,
        perim=1.0,
        shape3=shape,
        kind=candidate.kind,
        proposal_score=float(candidate.score),
    )


class HybridDRLQNSolver(DRLQNSolver):
    """Drop-in solver using dynamic spatial ACTION6 anchors.

    The parent class is deliberately reused for replay, Double-DQN updates,
    hard graph routing and differential runtime. Only perception/action proposal
    construction changes, making ablations straightforward.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._previous_grid: np.ndarray | None = None

    def perceive(self, env, grid) -> NodeInfo:
        node = super().perceive(env, grid)
        # A cached node already has the hybrid representation.
        if getattr(node, "hybrid_spatial", False):
            return node

        proposals = spatial_anchors(grid, node.objs, self._previous_grid)
        anchors = [_as_obj(p, grid) for p in proposals]
        if anchors:
            node.objs = anchors
            node.lam_feats = arg_features(anchors)
            node.valid = valid_mask(env, anchors)
        node.hybrid_spatial = True
        node.proposal_kinds = [p.kind for p in proposals]
        self._previous_grid = np.asarray(grid).copy()
        return node

    def _prior_bonus(self, node: NodeInfo, lam: int) -> float:
        bonus = super()._prior_bonus(node, lam)
        if lam >= 5:
            slot = (lam - 5) // 5
            if slot < len(node.objs):
                anchor = node.objs[slot]
                bonus += 0.08 * float(getattr(anchor, "proposal_score", 0.0))
                kind = getattr(anchor, "kind", "")
                if kind == "changed":
                    bonus += 1.0
                elif kind == "tile":
                    bonus += 0.35
                elif kind == "lattice":
                    bonus -= 0.15
        return bonus
