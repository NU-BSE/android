"""DRLQN model bundle: encoder + controller + Q-router + column router +
sparse residual lambda-executor."""
from __future__ import annotations

import numpy as np

from ..autodiff import T, Module
from ..config import DRLQNConfig
from .relational_encoder import RelationalEncoder
from .controller import Controller
from .q_router import QRouter
from .column_router import ColumnRouter
from .lambda_executor import LambdaExecutor


class DRLQN(Module):
    def __init__(self, cfg: DRLQNConfig, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.encoder = RelationalEncoder(cfg, rng)
        self.controller = Controller(cfg, rng)
        self.q_router = QRouter(cfg, rng)
        self.column_router = ColumnRouter(cfg, rng)
        self.executor = LambdaExecutor(cfg, rng)

    # --------------------------------------------------------------- hard path
    def encode(self, tokens, token_mask, trace) -> T:
        return self.encoder(tokens, token_mask, trace)

    def q_values(self, Z: T, route_ctx: T, valid: np.ndarray,
                 obj_feats: np.ndarray) -> T:
        I = self.controller(Z, route_ctx)
        return self.q_router(Z, I, route_ctx, valid, obj_feats)

    def route_and_execute(self, Z: T, route_ctx: T, lam_ids: np.ndarray,
                          lam_feats: np.ndarray):
        """Hard Top-k column routing, then the sparse residual executor.

        lam_feats (B, TOKEN_DIM): object-argument features per selected lambda.
        Returns (Z_plus, phi, margins, mask (B,C) bool, rho (B,C), gap).
        """
        cfg = self.cfg
        I = self.controller(Z, route_ctx)
        base = self.q_router.base_embeddings().gather_rows(lam_ids)   # (B, d_A)
        e_lam = base + self.q_router.obj_proj(T(lam_feats))           # (B, d_A)
        rho = self.column_router.scores(Z, I, e_lam)            # (B, C)
        mask, gap = ColumnRouter.topk_mask(rho.data, cfg.top_k)
        # gather active columns (same k for all batch rows)
        act_idx = np.stack([np.where(m)[0] for m in mask])      # (B, k)
        B = Z.shape[0]
        rows = np.repeat(np.arange(B), cfg.top_k)
        z_active = Z[rows, act_idx.reshape(-1)].reshape(B, cfg.top_k, cfg.d_c)
        phi, margins = self.executor(z_active, route_ctx, e_lam, lam_ids)
        # scatter residual back (Z+ = Z + M o Phi)
        M = np.zeros((B, cfg.C, 1))
        for b in range(B):
            M[b, act_idx[b], 0] = 1.0
        phi_full = self._scatter(phi, act_idx, B)
        Z_plus = Z + T(M) * phi_full
        return Z_plus, phi, margins, mask, rho, gap

    def _scatter(self, phi: T, act_idx: np.ndarray, B: int) -> T:
        cfg = self.cfg
        out = T(np.zeros((B, cfg.C, cfg.d_c)))
        # differentiable scatter via index assignment trick
        flat_idx = (np.arange(B)[:, None] * cfg.C + act_idx).reshape(-1)
        phi_flat = phi.reshape(B * cfg.top_k, cfg.d_c)
        base = T(np.zeros((B * cfg.C, cfg.d_c)))
        scat = _scatter_rows(base, flat_idx, phi_flat)
        return scat.reshape(B, cfg.C, cfg.d_c)


def _scatter_rows(base: T, idx: np.ndarray, rows: T) -> T:
    out = T(base.data.copy(), (rows,))
    np.add.at(out.data, idx, rows.data)

    def bw(g):
        return (g[idx],)

    out._backward = bw
    return out
