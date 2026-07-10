"""Q-router (spec 5.3): Double + Dueling DQN head with lambda-action
embeddings. e_k combines (a) a learned library embedding, (b) fixed operator
metadata features, and (c) for arity-1 click operators the token features of
the object argument the slot currently points at - this is what lets
compare-like operators transfer across states ("shared action geometry").
Invalid actions are masked to -inf before any argmax."""
from __future__ import annotations

import numpy as np

from ..autodiff import T, Param, Module, Linear, MLP
from ..config import DRLQNConfig
from ..operators.library import META_FEATURES, ARG_DIM


class QRouter(Module):
    def __init__(self, cfg: DRLQNConfig, rng):
        self.cfg = cfg
        d_u = cfg.d_c + cfg.d_i + cfg.d_route
        self.d_u = d_u
        # state summary: attention pooling over columns with a learned query
        self.pool_q = Param(rng.normal(0, 0.2, size=(cfg.d_c,)))
        self.u_proj = MLP([d_u, d_u], rng)
        # action embeddings: library + metadata (+ object argument features)
        self.E = Param(rng.normal(0, 0.2, size=(cfg.K, cfg.d_A)))
        self.e_proj = Linear(cfg.d_A + META_FEATURES.shape[1], cfg.d_A, rng)
        self.obj_proj = Linear(ARG_DIM, cfg.d_A, rng)
        # bilinear compatibility
        self.Wu = Linear(d_u, cfg.d_A, rng, bias=False)
        # relational term R_theta([P_u u; e_k])
        self.Pu = Linear(d_u, cfg.d_A, rng)
        self.R = MLP([2 * cfg.d_A, cfg.d_A, 1], rng)
        # dueling value head
        self.V = MLP([d_u, d_u, 1], rng)

    def base_embeddings(self) -> T:
        """(K, d_A) library embeddings with metadata, no object argument."""
        return self.e_proj(T.concat([self.E, T(META_FEATURES)], axis=-1))

    def effective_embeddings(self, obj_feats: np.ndarray) -> T:
        """obj_feats (B, K, ARG_DIM) -> (B, K, d_A)."""
        B = obj_feats.shape[0]
        base = self.base_embeddings().reshape(1, self.cfg.K, self.cfg.d_A) \
            + T(np.zeros((B, 1, 1)))
        return base + self.obj_proj(T(obj_feats))

    def effective_embedding_np(self, lam: int, lam_feat: np.ndarray) -> np.ndarray:
        """Numpy mirror for the runtime: single effective embedding."""
        e_in = np.concatenate([self.E.data[lam], META_FEATURES[lam]])
        base = e_in @ self.e_proj.W.data + self.e_proj.b.data
        return base + lam_feat @ self.obj_proj.W.data + self.obj_proj.b.data

    def summary(self, Z: T, I: T, route_ctx: T) -> T:
        B, C = Z.shape[0], Z.shape[1]
        scores = (Z @ self.pool_q.reshape(self.cfg.d_c, 1)).reshape(B, C)
        w = scores.softmax(axis=-1).reshape(B, C, 1)
        zp = (Z * w).sum(axis=1)                       # (B, d_c)
        ip = I.mean(axis=1)                            # (B, d_i)
        return self.u_proj(T.concat([zp, ip, route_ctx], axis=-1)).relu()

    def __call__(self, Z: T, I: T, route_ctx: T, valid: np.ndarray,
                 obj_feats: np.ndarray) -> T:
        """-> Q (B, K), invalid entries at -1e9."""
        B = Z.shape[0]
        K, dA = self.cfg.K, self.cfg.d_A
        u = self.summary(Z, I, route_ctx)              # (B, d_u)
        e = self.effective_embeddings(obj_feats)       # (B, K, d_A)
        wu = self.Wu(u).reshape(B, 1, dA)
        bil = (wu * e).sum(axis=-1) * (1.0 / np.sqrt(dA))   # (B, K)
        pu = self.Pu(u).reshape(B, 1, dA) + T(np.zeros((1, K, 1)))
        rel = self.R(T.concat([pu, e], axis=-1)).reshape(B, K)
        A = bil + rel
        # dueling: subtract mean advantage over the *valid* set Lambda(s)
        vm = valid.astype(np.float64)
        denom = np.maximum(vm.sum(axis=-1, keepdims=True), 1.0)
        A_masked = A * T(vm)
        A_centered = A - A_masked.sum(axis=-1, keepdims=True) / T(denom)
        Q = self.V(u) + A_centered
        return Q + T(np.where(valid, 0.0, -1e9))
