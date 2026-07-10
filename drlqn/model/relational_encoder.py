"""Relational encoder (spec 5.1): object tokens -> self-attention blocks ->
C learned slot queries -> logical columns Z in R^(C x d_c)."""
from __future__ import annotations

import numpy as np

from ..autodiff import T, Param, Module, Linear, MLP, LayerNorm
from ..config import DRLQNConfig
from ..state import TOKEN_DIM


class MultiHeadAttention(Module):
    def __init__(self, d, heads, rng):
        self.h = heads
        self.dk = d // heads
        self.Wq = Linear(d, d, rng)
        self.Wk = Linear(d, d, rng)
        self.Wv = Linear(d, d, rng)
        self.Wo = Linear(d, d, rng)

    def _split(self, x: T, B, n):
        # (B, n, d) -> (B, h, n, dk)
        return x.reshape(B, n, self.h, self.dk).transpose(0, 2, 1, 3)

    def __call__(self, q: T, kv: T, key_mask: np.ndarray | None = None) -> T:
        B, nq = q.shape[0], q.shape[1]
        nk = kv.shape[1]
        Q = self._split(self.Wq(q), B, nq)
        K = self._split(self.Wk(kv), B, nk)
        V = self._split(self.Wv(kv), B, nk)
        scores = (Q @ K.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.dk))
        if key_mask is not None:                      # (B, nk) True = valid
            bias = np.where(key_mask, 0.0, -1e9)[:, None, None, :]
            scores = scores + T(bias)
        att = scores.softmax(axis=-1)
        out = (att @ V).transpose(0, 2, 1, 3).reshape(B, nq, self.h * self.dk)
        return self.Wo(out)


class EncoderBlock(Module):
    def __init__(self, d, heads, rng):
        self.att = MultiHeadAttention(d, heads, rng)
        self.n1 = LayerNorm(d)
        self.n2 = LayerNorm(d)
        self.ff = MLP([d, 2 * d, d], rng)

    def __call__(self, x: T, key_mask) -> T:
        x = self.n1(x + self.att(x, x, key_mask))
        return self.n2(x + self.ff(x))


class RelationalEncoder(Module):
    def __init__(self, cfg: DRLQNConfig, rng, trace_dim: int = 4):
        d = cfg.enc_d
        self.cfg = cfg
        self.embed = Linear(TOKEN_DIM, d, rng)
        self.trace_embed = Linear(trace_dim, d, rng)
        self.blocks = [EncoderBlock(d, cfg.enc_heads, rng) for _ in range(cfg.enc_blocks)]
        self.slots = Param(rng.normal(0, 0.2, size=(cfg.C, d)))   # learned slot queries
        self.pool = MultiHeadAttention(d, cfg.enc_heads, rng)
        self.proj = Linear(d, cfg.d_c, rng)

    def __call__(self, tokens: np.ndarray, token_mask: np.ndarray,
                 trace: np.ndarray) -> T:
        """tokens (B, n, TOKEN_DIM); token_mask (B, n); trace (B, 4) -> Z (B, C, d_c)."""
        B = tokens.shape[0]
        x = self.embed(T(tokens))
        tr = self.trace_embed(T(trace)).reshape(B, 1, self.cfg.enc_d)
        x = T.concat([x, tr], axis=1)
        mask = np.concatenate([token_mask, np.ones((B, 1), dtype=bool)], axis=1)
        for blk in self.blocks:
            x = blk(x, mask)
        q = self.slots.reshape(1, self.cfg.C, self.cfg.enc_d) + T(np.zeros((B, 1, 1)))
        pooled = self.pool(q, x, mask)                 # (B, C, enc_d)
        return self.proj(pooled).tanh()                # (B, C, d_c)
