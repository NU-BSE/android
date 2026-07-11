"""Submission-facing top-route solver.

The exploratory ``TopRouteDRLQNSolver`` keeps detailed trace counters. For graph
identity, however, absolute step count must not be hashed: doing so would make
every visit unique and disable deterministic route reuse. This submission class
keys only the rendered state, level, and bounded recent action/effect context;
step count remains available to the neural encoder through trace features.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .top_solver import TopRouteDRLQNSolver


class SubmissionDRLQNSolver(TopRouteDRLQNSolver):
    """Top-route DRLQN with a genuinely bounded, revisitable history key."""

    def _context_key(self, grid: np.ndarray, level: int) -> str:
        payload = bytearray()
        for lam, changed, advanced in self._recent_effects:
            payload.extend(int(lam).to_bytes(2, "little", signed=False))
            payload.append(int(bool(changed)))
            payload.append(int(bool(advanced)))
        return hashlib.sha1(
            np.asarray(grid, dtype=np.uint8).tobytes()
            + int(level).to_bytes(2, "little", signed=False)
            + bytes(payload)
        ).hexdigest()[:16]
