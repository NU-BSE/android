"""DRLQN scale configurations (spec section 4.3)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DRLQNConfig:
    name: str = "DRLQN-M"
    C: int = 64            # logical columns
    d_c: int = 64          # column width  (E = C*d_c)
    d_i: int = 16          # control slot width (I = C*d_i)
    K: int = 128           # lambda library size
    d_A: int = 256         # action embedding dim
    top_k: int = 6         # hard column routing
    B: int = 8             # sparse residual executor depth
    d_b: int = 1024        # residual hidden
    rank: int = 16         # per-lambda low-rank adapter rank
    enc_d: int = 384       # relational encoder width
    enc_blocks: int = 6
    enc_heads: int = 4
    d_route: int = 16      # route context width
    # runtime guards (section 6)
    eta: float = 1.0
    margin_budget: float = 0.05
    router_budget: float = 0.02
    abs_tol: float = 0.05
    rel_tol: float = 1e-3
    audit_period: int = 32

    @property
    def d_E(self):
        return self.C * self.d_c

    @property
    def d_I(self):
        return self.C * self.d_i


def drlqn_m() -> DRLQNConfig:
    """Spec baseline: E=4096 (64x64), I=1024 (64x16), K=128, d_A=256, k=6."""
    return DRLQNConfig()


def drlqn_s() -> DRLQNConfig:
    """Spec DRLQN-S: B=6, d_b=512, r=8, encoder 256x4."""
    return DRLQNConfig(name="DRLQN-S", B=6, d_b=512, rank=8, enc_d=256, enc_blocks=4)


def drlqn_xs() -> DRLQNConfig:
    """CPU-trainable config for this sandbox (pure-numpy training).

    Deviation note: PyPI/torch is unavailable here and training runs on 4 CPU
    cores through a numpy autodiff engine, so the *trained* agent uses this
    reduced geometry. DRLQN-S/M are instantiated and smoke-tested at full
    spec scale (see tests/test_scale.py) and used for the hard-path forward
    checks; all architectural invariants are identical across scales.
    """
    return DRLQNConfig(name="DRLQN-XS", C=16, d_c=16, d_i=8, K=128, d_A=32,
                       top_k=4, B=2, d_b=96, rank=4, enc_d=48, enc_blocks=2,
                       enc_heads=2, d_route=8)
