"""Interneuron controller (spec 5.2): 16-d control slot per logical column.

i_c = G_psi(z_c, pool(neighbourhood), route_ctx). Feed-forward in v0; it does
not pick lambdas, it modulates the Q-head and the column router (E:I = 4:1
control prior).
"""
from __future__ import annotations

import numpy as np

from ..autodiff import T, Module, MLP
from ..config import DRLQNConfig


class Controller(Module):
    def __init__(self, cfg: DRLQNConfig, rng):
        self.cfg = cfg
        d_in = cfg.d_c + cfg.d_c + cfg.d_route
        self.mlp = MLP([d_in, 2 * cfg.d_i, cfg.d_i], rng)

    def __call__(self, Z: T, route_ctx: T) -> T:
        """Z (B, C, d_c), route_ctx (B, d_route) -> I (B, C, d_i)."""
        B, C = Z.shape[0], Z.shape[1]
        neigh = Z.mean(axis=1, keepdims=True) + T(np.zeros((1, C, 1)))    # (B,C,d_c)
        r = route_ctx.reshape(B, 1, self.cfg.d_route) + T(np.zeros((1, C, 1)))
        x = T.concat([Z, neigh, r], axis=-1)
        return self.mlp(x).tanh()
