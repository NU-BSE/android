"""Stability guards (spec 6.3 - 6.5). First-order updates are accepted only
while hard routing is provably stable in the local estimate; otherwise the
runtime falls back to exact recomputation. Soft quantities never decide
authoritative outcomes."""
from __future__ import annotations

import numpy as np


def branch_stable(m: float, dm: float, eta: float, eps_m: float) -> bool:
    """|m| > eta*|dm| + eps_m  (spec 6.3)."""
    return abs(m) > eta * abs(dm) + eps_m


def topk_stable(rho: np.ndarray, drho: np.ndarray, k: int, eps_router: float) -> bool:
    """rho_(k) - rho_(k+1) > 2*||drho||_inf + eps_router  (spec 6.4)."""
    if k >= rho.shape[0]:
        return True
    srt = -np.sort(-rho)
    gap = srt[k - 1] - srt[k]
    return gap > 2.0 * float(np.abs(drho).max()) + eps_router


def linear_stable(remainder_ema: float, dz: np.ndarray, z: np.ndarray,
                  abs_tol: float, rel_tol: float) -> bool:
    """Predicted second-order error e_hat*||dz||^2 < tau_abs + tau_rel*||z|| (spec 6.5)."""
    err = remainder_ema * float((dz ** 2).sum())
    return err < abs_tol + rel_tol * float(np.linalg.norm(z))


def measured_remainder(phi_new: np.ndarray, phi_old: np.ndarray,
                       jvp_pred: np.ndarray, dz: np.ndarray) -> float:
    """e = ||Phi(z+dz) - Phi(z) - DPhi(z)dz|| / (||dz||^2 + eps)  (spec 6.5)."""
    num = float(np.linalg.norm(phi_new - phi_old - jvp_pred))
    return num / (float((dz ** 2).sum()) + 1e-8)
