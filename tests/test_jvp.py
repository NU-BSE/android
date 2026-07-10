import numpy as np

from drlqn.config import drlqn_xs
from drlqn.model.drlqn import DRLQN
from drlqn.runtime.jvp import (jvp_controller, jvp_column_scores, jvp_executor,
                               exact_executor)


def test_jvp_matches_finite_difference():
    cfg = drlqn_xs()
    model = DRLQN(cfg, seed=1)
    rng = np.random.default_rng(0)
    C, dc = cfg.C, cfg.d_c
    Z = rng.normal(0, 0.5, size=(C, dc))
    dZ = rng.normal(0, 1.0, size=(C, dc))
    r = rng.normal(0, 0.5, size=(cfg.d_route,))
    e = model.q_router.effective_embedding_np(3, np.zeros(38))
    eps = 1e-6

    # controller
    I, dI = jvp_controller(model.controller, Z, dZ, r)
    I2, _ = jvp_controller(model.controller, Z + eps * dZ, np.zeros_like(dZ), r)
    fd = (I2 - I) / eps
    assert np.abs(fd - dI).max() < 1e-4

    # column router
    rho, drho = jvp_column_scores(model.column_router, Z, dZ, I, dI, e)
    I2, _ = jvp_controller(model.controller, Z + eps * dZ, np.zeros_like(dZ), r)
    rho2, _ = jvp_column_scores(model.column_router, Z + eps * dZ,
                                np.zeros_like(dZ), I2, np.zeros_like(dI), e)
    fd = (rho2 - rho) / eps
    assert np.abs(fd - drho).max() < 1e-4

    # executor
    k = cfg.top_k
    z, dz = Z[:k], dZ[:k]
    phi, dphi, m, dm = jvp_executor(model.executor, z, dz, r, e, 3)
    phi2, m2 = exact_executor(model.executor, z + eps * dz, r, e, 3)
    assert np.abs((phi2 - phi) / eps - dphi).max() < 1e-3
    assert np.abs((m2 - m) / eps - dm).max() < 1e-3


def test_numpy_mirror_matches_engine_forward():
    """exact_executor must reproduce the autodiff-engine executor bitwise-ish."""
    from drlqn.autodiff import T
    cfg = drlqn_xs()
    model = DRLQN(cfg, seed=2)
    rng = np.random.default_rng(1)
    k = cfg.top_k
    z = rng.normal(0, 0.5, size=(1, k, cfg.d_c))
    r = rng.normal(0, 0.5, size=(1, cfg.d_route))
    lam = np.array([7])
    e_np = model.q_router.effective_embedding_np(7, np.zeros(38))
    phi_engine, m_engine = model.executor(T(z), T(r), T(e_np[None]), lam)
    phi_np, m_np = exact_executor(model.executor, z[0], r[0], e_np, 7)
    assert np.abs(phi_engine.data[0] - phi_np).max() < 1e-10
    assert np.abs(m_engine.data[0] - m_np).max() < 1e-10


if __name__ == "__main__":
    test_jvp_matches_finite_difference()
    test_numpy_mirror_matches_engine_forward()
    print("jvp checks OK")
