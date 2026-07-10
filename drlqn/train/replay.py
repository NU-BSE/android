"""Replay buffer (spec 7.6): stores canonical state handles (frame keys), not
latents. Latents are re-encoded per update so replay never goes stale; an
encoder_version hash guards any cached Z reuse at act time."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Transition:
    key: str
    lam: int
    reward: float
    next_key: str
    done: bool
    stable_pair: bool      # valid semantic delta, same level, frame changed
    changed: bool          # did the frame change at all


class Replay:
    def __init__(self, capacity: int = 20000, seed: int = 0):
        self.buf: list[Transition] = []
        self.capacity = capacity
        self.rng = random.Random(seed)

    def push(self, tr: Transition):
        if len(self.buf) >= self.capacity:
            self.buf.pop(0)
        self.buf.append(tr)

    def sample(self, n: int) -> list[Transition]:
        return self.rng.sample(self.buf, min(n, len(self.buf)))

    def __len__(self):
        return len(self.buf)
