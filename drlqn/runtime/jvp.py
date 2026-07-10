"""Hand-written forward-mode JVP for the DRLQN operator graph.

The spec prototypes this with torch.func.jvp; torch cannot be installed in
this sandbox (PyPI blocked), and the executor/controller/router graphs are
small and fixed, so exact forward-mode tangents are written out by hand and
verified against finite differences in tests/test_jvp.py. No full Jacobian
is ever materialised.
"""
from __future__ import annotations

import numpy as np

from ..model.controller import Controller
from ..model.column_router import ColumnRouter
from ..model.lambda_executor import LambdaExecutor


# ------------------------------------------------------------------ primitives
def _lin(layer, x, dx):
    W, b = layer.W.data, (layer.b.data if layer.b is not None else 0.0)
    return x @ W + b, dx @ W


def _relu(y, dy):
    m = y > 0
    return y * m, dy * m


def _tanh(y, dy):
    t = np.tanh(y)
    return t, dy * (1 - t * t)


def _mlp(mlp, x, dx):
    for i, layer in enumerate(mlp.layers):
        x, dx = _lin(layer, x, dx)
        if i < len(mlp.layers) - 1:
            x, dx = _relu(x, dx) if mlp.act == "relu" else _tanh(x, dx)
    return x, dx


# ----------------------------------------------------------------- controller
def jvp_controller(ctrl: Controller, Z, dZ, route_ctx):
    """Z (C,d_c), dZ tangent, route_ctx (d_route) -> (I, dI) each (C,d_i)."""
    C = Z.shape[0]
    neigh = np.broadcast_to(Z.mean(axis=0, keepdims=True), Z.shape)
    dneigh = np.broadcast_to(dZ.mean(axis=0, keepdims=True), Z.shape)
    r = np.broadcast_to(route_ctx[None, :], (C, route_ctx.shape[0]))
    x = np.concatenate([Z, neigh, r], axis=-1)
    dx = np.concatenate([dZ, dneigh, np.zeros_like(r)], axis=-1)
    y, dy = _mlp(ctrl.mlp, x, dx)
    return _tanh(y, dy)


# --------------------------------------------------------------- column router
def jvp_column_scores(router: ColumnRouter, Z, dZ, I, dI, e_lam):
    """-> (rho (C,), drho (C,))."""
    a = e_lam @ router.Wa.W.data + router.Wa.b.data
    pre = (Z @ router.Wz.W.data + router.Wz.b.data
           + I @ router.Wi.W.data + router.Wi.b.data + a[None, :])
    dpre = dZ @ router.Wz.W.data + dI @ router.Wi.W.data
    h, dh = _tanh(pre, dpre)
    w = router.w.data[:, 0]
    return h @ w, dh @ w


# ------------------------------------------------------------------- executor
def jvp_executor(execu: LambdaExecutor, z_active, dz_active, route_ctx, e_lam,
                 lam_id: int):
    """z_active (k,d_c) -> (phi, dphi, margins, dmargins)."""
    k = z_active.shape[0]
    z, dz = z_active.copy(), dz_active.copy()
    for blk in execu.blocks:
        neigh = np.broadcast_to(z.mean(axis=0, keepdims=True), z.shape)
        dneigh = np.broadcast_to(dz.mean(axis=0, keepdims=True), z.shape)
        r = np.broadcast_to(route_ctx[None, :], (k, route_ctx.shape[0]))
        x = np.concatenate([z, neigh, r], axis=-1)
        dx = np.concatenate([dz, dneigh, np.zeros_like(r)], axis=-1)
        h, dh = _lin(blk.W1, x, dx)
        h, dh = _relu(h, dh)
        A, Bm = blk.A.data[lam_id], blk.Bm.data[lam_id]
        h_lam = h + (h @ A) @ Bm
        dh_lam = dh + (dh @ A) @ Bm
        g = 1.0 + (e_lam @ blk.film_g.W.data + blk.film_g.b.data)
        b = e_lam @ blk.film_b.W.data + blk.film_b.b.data
        pre = g[None, :] * h_lam + b[None, :]
        dpre = g[None, :] * dh_lam
        act, dact = _relu(pre, dpre)
        dz_blk, ddz_blk = _lin(blk.W2, act, dact)
        z = z + dz_blk
        dz = dz + ddz_blk
    phi, dphi = z - z_active, dz - dz_active
    m_in = np.concatenate([z.mean(axis=0), route_ctx])
    dm_in = np.concatenate([dz.mean(axis=0), np.zeros_like(route_ctx)])
    m, dm = _mlp(execu.margin, m_in, dm_in)
    return phi, dphi, m, dm


def jvp_q_value(model, Z, dZ, route_ctx, e_lam: np.ndarray):
    """Forward-mode J_Z Q(s, lambda) . dZ for one lambda (effective embedding
    e_lam is constant w.r.t. Z). -> (q, dq)."""
    qr = model.q_router
    I, dI = jvp_controller(model.controller, Z, dZ, route_ctx)
    # attention pooling over columns
    scores = Z @ qr.pool_q.data
    dscores = dZ @ qr.pool_q.data
    e = np.exp(scores - scores.max())
    s = e / e.sum()
    ds = s * (dscores - float((s * dscores).sum()))
    zp = s @ Z
    dzp = ds @ Z + s @ dZ
    ip, dip = I.mean(axis=0), dI.mean(axis=0)
    x = np.concatenate([zp, ip, route_ctx])
    dx = np.concatenate([dzp, dip, np.zeros_like(route_ctx)])
    u, du = _mlp(qr.u_proj, x, dx)
    u, du = _relu(u, du)
    ek = e_lam
    dA = model.cfg.d_A
    wu, dwu = _lin(qr.Wu, u, du)
    bil, dbil = float(wu @ ek) / np.sqrt(dA), float(dwu @ ek) / np.sqrt(dA)
    pu, dpu = _lin(qr.Pu, u, du)
    rel_in = np.concatenate([pu, ek])
    drel_in = np.concatenate([dpu, np.zeros_like(ek)])
    rel, drel = _mlp(qr.R, rel_in, drel_in)
    v, dv = _mlp(qr.V, u, du)
    # NOTE: single-lambda view; the dueling mean-advantage shift is common to
    # all lambdas and cancels in DeltaQ comparisons between the same state pair.
    q = float(v[0] + bil + rel[0])
    dq = float(dv[0] + dbil + drel[0])
    return q, dq


def exact_executor(execu: LambdaExecutor, z_active, route_ctx, e_lam, lam_id: int):
    """Exact primal forward (numpy mirror, no tangent)."""
    zeros = np.zeros_like(z_active)
    phi, _, m, _ = jvp_executor(execu, z_active, zeros, route_ctx, e_lam, lam_id)
    return phi, m
