"""Training losses (spec section 7).

L = L_Q + mu_Q L_dQ + mu_op L_op + mu_R L_router  (+ world-model auxiliaries
that give the executor its latent-dynamics semantics and train the margins).

Stop-gradient note: the JVP targets in L_dQ and L_op are computed by the
hand-written forward-mode pass and enter the loss as constants; gradients
flow through the exact primal terms only. This keeps reverse-over-forward
out of the numpy engine and matches the "differential shadow is advisory"
invariant.
"""
from __future__ import annotations

import numpy as np

from ..autodiff import T, huber
from ..runtime.jvp import jvp_q_value, jvp_executor


def double_dqn_targets(model, target_model, batch_next, valid_next, rewards,
                       dones, gamma: float, obj_feats_next: np.ndarray) -> np.ndarray:
    """y = r + gamma (1-d) Q_theta-(s', argmax_lambda Q_theta(s', lambda))."""
    tokens, mask, trace = batch_next
    Z_on = model.encode(tokens, mask, trace)
    q_on = model.q_values(Z_on, T(np.zeros((tokens.shape[0], model.cfg.d_route))),
                          valid_next, obj_feats_next).data
    lam_star = q_on.argmax(axis=-1)
    Z_tg = target_model.encode(tokens, mask, trace)
    q_tg = target_model.q_values(Z_tg, T(np.zeros((tokens.shape[0], model.cfg.d_route))),
                                 valid_next, obj_feats_next).data
    q_next = q_tg[np.arange(len(lam_star)), lam_star]
    q_next = np.where(valid_next.any(axis=-1), q_next, 0.0)
    return rewards + gamma * (1.0 - dones) * q_next


def q_loss(model, Z: T, route_ctx: T, valid: np.ndarray, lams: np.ndarray,
           targets: np.ndarray, obj_feats: np.ndarray) -> T:
    q = model.q_values(Z, route_ctx, valid, obj_feats)
    q_sel = q[np.arange(len(lams)), lams]
    return huber(q_sel - T(targets))


def world_model_losses(model, Z: T, route_ctx: T, lams: np.ndarray,
                       lam_feats: np.ndarray, Z_next_sg: np.ndarray,
                       changed: np.ndarray, same_level: np.ndarray):
    """Executor as latent dynamics + margin as change predictor.

    L_model = ||Z + M o Phi - sg(Z_next)||^2   (stable pairs only)
    L_margin = softplus(-y * m0), y = +1 if frame changed else -1
    """
    Z_plus, phi, margins, mask, rho, gap = model.route_and_execute(Z, route_ctx, lams, lam_feats)
    w = same_level.astype(np.float64)[:, None, None]
    diff = (Z_plus - T(Z_next_sg)) * T(w)
    l_model = (diff * diff).mean()
    y = np.where(changed, 1.0, -1.0)
    m0 = margins[:, 0]
    # softplus(-y m) via engine ops
    zx = m0 * T(-y)
    l_margin = ((zx.exp() + 1.0).log()).mean()
    # weak router load-balancing warm-up (spec 5.4): encourage usage entropy
    p = rho.softmax(axis=-1).mean(axis=0)
    l_router = (p * (p + 1e-8).log()).sum()
    return l_model, l_margin, l_router, margins, gap


def differential_q_loss(model, nodes_Z: list[np.ndarray], dZs: list[np.ndarray],
                        route_ctx: np.ndarray, lams: list[int],
                        e_lams: list[np.ndarray], lam_feats: list[np.ndarray],
                        Z_pairs: list[tuple[T, T]]) -> T:
    """L_dQ = Huber(DeltaQ - J_Z Q . dZ) on route-stable pairs (spec 7.2)."""
    losses = []
    for Zt, dZ, lam, e_lam, lf, (Za, Zb) in zip(nodes_Z, dZs, lams, e_lams,
                                                lam_feats, Z_pairs):
        _, dq = jvp_q_value(model, Zt, dZ, route_ctx, e_lam)  # stop-grad target
        qa = _q_single(model, Za, lam, lf)
        qb = _q_single(model, Zb, lam, lf)
        losses.append(huber((qb - qa) - T(np.array(dq))))
    if not losses:
        return T(np.array(0.0))
    total = losses[0]
    for l in losses[1:]:
        total = total + l
    return total / float(len(losses))


def _q_single(model, Z: T, lam: int, lam_feat: np.ndarray) -> T:
    B = Z.shape[0]
    valid = np.zeros((B, model.cfg.K), dtype=bool)
    valid[:, lam] = True
    obj_feats = np.zeros((B, model.cfg.K, lam_feat.shape[-1]))
    obj_feats[:, lam] = lam_feat
    q = model.q_values(Z, T(np.zeros((B, model.cfg.d_route))), valid, obj_feats)
    return q[np.arange(B), np.full(B, lam)]


def operator_linearity_loss(model, Z_active: T, dz: np.ndarray,
                            route_ctx: np.ndarray, e_lam, lam: int) -> T:
    """L_op = ||Phi(Z+dZ) - Phi(Z) - sg(DPhi dZ)||^2 (spec 7.3)."""
    _, dphi, _, _ = jvp_executor(model.executor, Z_active.data[0], dz,
                                 route_ctx, e_lam.data[0], lam)
    lam_ids = np.array([lam])
    rc = T(route_ctx[None])
    phi_a, _ = model.executor(Z_active, rc, e_lam, lam_ids)
    phi_b, _ = model.executor(Z_active + T(dz[None]), rc, e_lam, lam_ids)
    diff = phi_b - phi_a - T(dphi[None])
    return (diff * diff).mean()
