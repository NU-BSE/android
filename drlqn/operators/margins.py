"""RADDT-style path violations and the softmin differential shadow (spec 2.2).

The margin m produced by the executor predicts whether a lambda will change
the frame (branch b = 1[m >= 0]). For multiway route candidates we build
U_t(s) = sum_j ReLU(-sigma_tj m_j) and its softmin shadow p_t; hard argmin is
authoritative, the shadow only reports which alternative routes become close
under a tangent (advisory only - never an environment transition).
"""
from __future__ import annotations

import numpy as np


def path_violation(margins: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """U_t = sum_j ReLU(-sigma_tj * m_j).  margins: (J,), signs: (T, J)."""
    v = np.maximum(0.0, -signs * margins[None, :])
    return v.sum(axis=1)


def hard_route(U: np.ndarray) -> int:
    return int(np.argmin(U))


def soft_shadow(U: np.ndarray, alpha: float = 4.0) -> np.ndarray:
    z = -alpha * (U - U.min())
    e = np.exp(z)
    return e / e.sum()


def shadow_differential(p: np.ndarray, dU: np.ndarray, alpha: float = 4.0) -> np.ndarray:
    """delta p_t = -alpha p_t (dU_t - sum_q p_q dU_q)."""
    return -alpha * p * (dU - float((p * dU).sum()))
