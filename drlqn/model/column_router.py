"""Column router (spec 5.4): hard Top-k selection of logical columns for the
selected lambda. rho_c = w^T tanh(Wz z_c + Wi i_c + Wa e_lambda + b)."""
from __future__ import annotations

import numpy as np

from ..autodiff import T, Param, Module, Linear
from ..config import DRLQNConfig


class ColumnRouter(Module):
    def __init__(self, cfg: DRLQNConfig, rng):
        self.cfg = cfg
        self.Wz = Linear(cfg.d_c, cfg.d_c, rng)
        self.Wi = Linear(cfg.d_i, cfg.d_c, rng)
        self.Wa = Linear(cfg.d_A, cfg.d_c, rng)
        self.w = Param(rng.normal(0, 0.3, size=(cfg.d_c, 1)))

    def scores(self, Z: T, I: T, e_lam: T) -> T:
        """Z (B,C,d_c), I (B,C,d_i), e_lam (B,d_A) -> rho (B, C)."""
        B, C = Z.shape[0], Z.shape[1]
        a = self.Wa(e_lam).reshape(B, 1, self.cfg.d_c) + T(np.zeros((1, C, 1)))
        h = (self.Wz(Z) + self.Wi(I) + a).tanh()
        return (h @ self.w).reshape(B, C)

    @staticmethod
    def topk_mask(rho: np.ndarray, k: int):
        """Hard Top-k (authoritative). Returns (mask (B,C) bool, gap (B,))."""
        B, C = rho.shape
        order = np.argsort(-rho, axis=-1)
        mask = np.zeros((B, C), dtype=bool)
        np.put_along_axis(mask, order[:, :k], True, axis=-1)
        srt = -np.sort(-rho, axis=-1)
        gap = srt[:, k - 1] - srt[:, k] if k < C else np.full(B, np.inf)
        return mask, gap
