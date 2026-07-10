"""Differential shadow (advisory only). Computes path violations U_t over
route candidates from executor margins, the softmin weights p_t and their
first-order response dp_t to the current tangent. Consumed for
sensitivity / invalidation diagnostics - never for authoritative routing."""
from __future__ import annotations

import numpy as np

from ..operators.margins import path_violation, soft_shadow, shadow_differential

# two-branch template per margin: sigma = +1 (branch taken) / -1 (not taken)
_SIGNS = np.array([[+1.0], [-1.0]])


def shadow_report(m: np.ndarray, dm: np.ndarray, alpha: float = 4.0) -> dict:
    """m, dm: margin vector and its JVP. Builds U_t, p_t, dp_t per margin."""
    report = {}
    for j in range(m.shape[0]):
        U = path_violation(m[j:j + 1], _SIGNS)
        dU = np.array([
            max(0.0, -m[j]) and -dm[j] or 0.0,
            max(0.0, m[j]) and dm[j] or 0.0,
        ])
        p = soft_shadow(U, alpha)
        dp = shadow_differential(p, dU, alpha)
        report[j] = {"U": U, "p": p, "dp": dp,
                     "flip_risk": float(p[1 - int(np.argmin(U))])}
    return report
