"""NodeCache (spec 6.1 / 8.4). Runtime cache only - never part of the MDP
state, never allowed to influence rewards."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NodeCache:
    Z: np.ndarray                        # (C, d_c) primal latent
    I: np.ndarray | None                 # (C, d_i) controller output
    lambda_id: int                       # operator used at this node (-1 root)
    active_indices: np.ndarray | None    # Top-k column ids
    rho: np.ndarray | None               # (C,) routing scores
    cached_residual: np.ndarray | None   # (k, d_c) Phi_lambda on active columns
    margin: np.ndarray | None            # (N_MARGINS,)
    path_violation: np.ndarray | None = None
    shadow_weights: np.ndarray | None = None
    encoder_version: int = 0
    accepted_jvp: int = 0
    is_exact: bool = True                # False if Z came from approximate update


@dataclass
class RemainderEMA:
    """Measured Taylor remainder scale per operator (spec 6.5)."""
    value: dict = field(default_factory=dict)
    beta: float = 0.9

    def update(self, lam: int, e: float):
        v = self.value.get(lam, e)
        self.value[lam] = self.beta * v + (1 - self.beta) * e

    def get(self, lam: int, default: float = 1.0) -> float:
        return self.value.get(lam, default)


@dataclass
class RuntimeTelemetry:
    jvp_accepted: int = 0
    exact_recompute: int = 0
    audits: int = 0
    audit_drift: list = field(default_factory=list)
    false_stable: int = 0                # severity-1: guard said stable, audit disagreed
    guard_fail_branch: int = 0
    guard_fail_topk: int = 0
    guard_fail_linear: int = 0
    encoder_calls: int = 0
    encoder_calls_avoided: int = 0

    def summary(self) -> dict:
        tot = self.jvp_accepted + self.exact_recompute
        return {
            "jvp_accepted": self.jvp_accepted,
            "exact_recompute": self.exact_recompute,
            "jvp_acceptance_rate": self.jvp_accepted / tot if tot else 0.0,
            "encoder_calls": self.encoder_calls,
            "encoder_calls_avoided": self.encoder_calls_avoided,
            "exact_forward_avoidance": (
                self.encoder_calls_avoided /
                (self.encoder_calls + self.encoder_calls_avoided)
                if (self.encoder_calls + self.encoder_calls_avoided) else 0.0),
            "audits": self.audits,
            "mean_audit_drift": float(np.mean(self.audit_drift)) if self.audit_drift else 0.0,
            "max_audit_drift": float(np.max(self.audit_drift)) if self.audit_drift else 0.0,
            "false_stable": self.false_stable,
            "guard_fail_branch": self.guard_fail_branch,
            "guard_fail_topk": self.guard_fail_topk,
            "guard_fail_linear": self.guard_fail_linear,
        }
