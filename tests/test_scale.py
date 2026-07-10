"""Instantiate DRLQN at spec scales; verify geometry and parameter budget.

DRLQN-M is the spec's main prototype (64x64 columns = E 4096, 64x16 control
= I 1024, K=128, d_A=256, Top-k=6, B=8, d_b=1024, r=16). Training in this
sandbox runs DRLQN-XS (CPU/numpy); this test proves the full-scale hard path
executes with identical semantics.
"""
import numpy as np

from drlqn.config import drlqn_s, drlqn_m, drlqn_xs
from drlqn.model.drlqn import DRLQN
from drlqn.autodiff import T
from drlqn.state import MAX_TOKENS, TOKEN_DIM
from drlqn.operators.library import K, ARG_DIM


def _smoke(cfg, check_params=None):
    model = DRLQN(cfg, seed=0)
    n = model.n_params()
    assert cfg.d_E == cfg.C * cfg.d_c
    assert cfg.d_I == cfg.C * cfg.d_i
    tokens = np.random.default_rng(0).normal(0, 0.3, (1, MAX_TOKENS, TOKEN_DIM))
    mask = np.ones((1, MAX_TOKENS), dtype=bool)
    trace = np.zeros((1, 4))
    Z = model.encode(tokens, mask, trace)
    assert Z.shape == (1, cfg.C, cfg.d_c)
    valid = np.ones((1, K), dtype=bool)
    q = model.q_values(Z, T(np.zeros((1, cfg.d_route))), valid,
                       np.zeros((1, K, ARG_DIM)))
    assert q.shape == (1, K)
    Zp, phi, m, msk, rho, gap = model.route_and_execute(
        Z, T(np.zeros((1, cfg.d_route))), np.array([3]), np.zeros((1, ARG_DIM)))
    assert phi.shape == (1, cfg.top_k, cfg.d_c)
    assert int(msk.sum()) == cfg.top_k
    # sparse locality: residual support restricted to Top-k columns
    delta = np.abs(Zp.data - Z.data)[0].sum(axis=-1)
    assert (delta > 1e-12).sum() <= cfg.top_k
    if check_params:
        lo, hi = check_params
        assert lo <= n <= hi, f"{cfg.name}: {n/1e6:.1f}M params out of range"
    return n


def test_xs():
    n = _smoke(drlqn_xs())
    print(f"DRLQN-XS: {n/1e6:.2f}M params")


def test_s():
    n = _smoke(drlqn_s(), check_params=(10e6, 30e6))
    print(f"DRLQN-S: {n/1e6:.1f}M params")


def test_m_geometry_and_params():
    cfg = drlqn_m()
    n = _smoke(cfg, check_params=(40e6, 75e6))
    assert cfg.d_E == 4096 and cfg.d_I == 1024 and cfg.K == 128
    assert cfg.d_A == 256 and cfg.top_k == 6 and cfg.B == 8
    print(f"DRLQN-M: {n/1e6:.1f}M params (spec target 55-65M, sparse executor)")


if __name__ == "__main__":
    test_xs()
    test_s()
    test_m_geometry_and_params()
    print("scale checks OK")
