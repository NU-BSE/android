"""Sparse residual lambda-executor (spec 5.5).

B residual blocks run only on the Top-k active columns. Each block has a
shared base and a per-lambda low-rank adapter in hidden space plus FiLM
modulation from the action embedding:

    h      = relu(W1 [z_c; neigh_c; route_ctx])
    h_lam  = h + (h A_lam) B_lam
    dz_c   = relu(gamma(e) * h_lam + beta(e)) W2
    z_c   <- z_c + dz_c

The executor returns the total residual Phi_lambda(Z) on active columns and
branch margins m (m[0] = "this lambda changes the frame" margin; m[1:] are
route margins for path-violation scoring). The margin head reads only active
columns + route context (locality invariant).
"""
from __future__ import annotations

import numpy as np

from ..autodiff import T, Param, Module, Linear, MLP
from ..config import DRLQNConfig

N_MARGINS = 4


class ExecutorBlock(Module):
    def __init__(self, cfg: DRLQNConfig, rng):
        self.cfg = cfg
        d_x = 2 * cfg.d_c + cfg.d_route
        self.W1 = Linear(d_x, cfg.d_b, rng)
        self.W2 = Linear(cfg.d_b, cfg.d_c, rng, scale=0.02)
        self.A = Param(rng.normal(0, 0.05, size=(cfg.K, cfg.d_b, cfg.rank)))
        self.Bm = Param(rng.normal(0, 0.05, size=(cfg.K, cfg.rank, cfg.d_b)))
        self.film_g = Linear(cfg.d_A, cfg.d_b, rng, scale=0.05)
        self.film_b = Linear(cfg.d_A, cfg.d_b, rng, scale=0.05)

    def __call__(self, z: T, route_ctx: T, e_lam: T, lam_ids: np.ndarray) -> T:
        """z (B,k,d_c) active columns -> dz (B,k,d_c)."""
        B, k = z.shape[0], z.shape[1]
        neigh = z.mean(axis=1, keepdims=True) + T(np.zeros((1, k, 1)))
        r = route_ctx.reshape(B, 1, self.cfg.d_route) + T(np.zeros((1, k, 1)))
        x = T.concat([z, neigh, r], axis=-1)
        h = self.W1(x).relu()                                   # (B,k,d_b)
        A = self.A.gather_rows(lam_ids)                         # (B,d_b,r)
        Bm = self.Bm.gather_rows(lam_ids)                       # (B,r,d_b)
        h_lam = h + (h @ A) @ Bm
        g = 1.0 + self.film_g(e_lam).reshape(B, 1, self.cfg.d_b)
        b = self.film_b(e_lam).reshape(B, 1, self.cfg.d_b)
        return self.W2((g * h_lam + b).relu())


class LambdaExecutor(Module):
    def __init__(self, cfg: DRLQNConfig, rng):
        self.cfg = cfg
        self.blocks = [ExecutorBlock(cfg, rng) for _ in range(cfg.B)]
        self.margin = MLP([cfg.d_c + cfg.d_route, cfg.d_c, N_MARGINS], rng)

    def __call__(self, z_active: T, route_ctx: T, e_lam: T, lam_ids: np.ndarray):
        """-> (Phi (B,k,d_c) residual, margins (B, N_MARGINS))."""
        z = z_active
        for blk in self.blocks:
            z = z + blk(z, route_ctx, e_lam, lam_ids)
        phi = z - z_active
        m_in = T.concat([z.mean(axis=1), route_ctx], axis=-1)
        return phi, self.margin(m_in)
