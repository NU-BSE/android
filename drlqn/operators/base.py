"""Lambda-operator contract (spec section 8.2).

In the ARC-AGI-3 instantiation a lambda-action is a typed environment
operator: five key operators plus click operators addressed to object slots.
The residual executor (model/lambda_executor.py) provides the differentiable
Phi_lambda used by the shadow; the authoritative transition is always the
exact environment step (invariant: hard semantics is authoritative).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorMeta:
    operator_id: int
    arity: int
    branch_count: int
    kind: str              # move | key | click
    mutates_relations: bool

    def features(self):
        """Fixed categorical metadata features fed next to the learned embedding."""
        import numpy as np
        f = np.zeros(8)
        f[0] = 1.0 if self.kind == "move" else 0.0
        f[1] = 1.0 if self.kind == "key" else 0.0
        f[2] = 1.0 if self.kind == "click" else 0.0
        f[3] = self.arity / 2.0
        f[4] = self.branch_count / 4.0
        f[5] = 1.0 if self.mutates_relations else 0.0
        f[6] = (self.operator_id % 16) / 16.0
        f[7] = (self.operator_id // 16) / 8.0
        return f
