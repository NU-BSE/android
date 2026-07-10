"""Approximate recompute runtime (spec 6.7): optimistic JVP, conservative
correction, periodic exact audit with re-anchor.

The exact path re-encodes the frame through the full relational encoder
(the authoritative primal). The approximate path advances the cached latent
with the residual world-model update Z+ = Z + M o Phi_lambda(Z), guarded by
the branch / Top-k / remainder stability tests; the tangent carried between
nodes is the last accepted residual. Guards failing => exact recompute.
"""
from __future__ import annotations

import numpy as np

from .cache import NodeCache, RemainderEMA, RuntimeTelemetry
from .jvp import jvp_controller, jvp_column_scores, jvp_executor
from .stability import (branch_stable, topk_stable, linear_stable,
                        measured_remainder)


class DifferentialRuntime:
    def __init__(self, model, cfg, telemetry: RuntimeTelemetry | None = None):
        self.model = model
        self.cfg = cfg
        self.remainder = RemainderEMA()
        self.tel = telemetry or RuntimeTelemetry()
        self.encoder_version = 0

    # ------------------------------------------------------------- exact path
    def exact_encode(self, tokens, token_mask, trace) -> NodeCache:
        Z = self.model.encode(tokens[None], token_mask[None], trace[None]).data[0]
        self.tel.encoder_calls += 1
        return NodeCache(Z=Z, I=None, lambda_id=-1, active_indices=None,
                         rho=None, cached_residual=None, margin=None,
                         encoder_version=self.encoder_version, is_exact=True)

    # -------------------------------------------------------- approximate path
    def apply_delta(self, node: NodeCache, lam_id: int, dz: np.ndarray,
                    route_ctx: np.ndarray, e_lam: np.ndarray,
                    exact_encode_fn) -> tuple[NodeCache, bool]:
        """Advance the cached latent for operator lam_id under tangent dz.

        e_lam: effective action embedding (library + metadata + object arg).
        Returns (next_node, used_approx). exact_encode_fn() must return the
        authoritative NodeCache built from the real next frame (exact
        fallback / audit target).
        """
        cfg = self.cfg
        model = self.model

        I, dI = jvp_controller(model.controller, node.Z, dz, route_ctx)
        rho, drho = jvp_column_scores(model.column_router, node.Z, dz, I, dI, e_lam)
        order = np.argsort(-rho)
        act = np.sort(order[:cfg.top_k])
        z_active, dz_active = node.Z[act], dz[act]
        phi, dphi, m, dm = jvp_executor(model.executor, z_active, dz_active,
                                        route_ctx, e_lam, lam_id)

        b_ok = branch_stable(float(m[0]), float(dm[0]), cfg.eta, cfg.margin_budget)
        k_ok = topk_stable(rho, drho, cfg.top_k, cfg.router_budget)
        l_ok = linear_stable(self.remainder.get(lam_id), dz_active, node.Z,
                             cfg.abs_tol, cfg.rel_tol)
        if not b_ok:
            self.tel.guard_fail_branch += 1
        if not k_ok:
            self.tel.guard_fail_topk += 1
        if not l_ok:
            self.tel.guard_fail_linear += 1

        if b_ok and k_ok and l_ok:
            Z_plus = node.Z.copy()
            Z_plus[act] += phi + dphi
            nxt = NodeCache(Z=Z_plus, I=I, lambda_id=lam_id, active_indices=act,
                            rho=rho, cached_residual=phi, margin=m,
                            encoder_version=node.encoder_version,
                            accepted_jvp=node.accepted_jvp + 1, is_exact=False)
            self.tel.jvp_accepted += 1
            self.tel.encoder_calls_avoided += 1
            # remainder audit on the fly: measure Taylor remainder of executor
            phi2, _, _, _ = jvp_executor(model.executor, z_active + dz_active,
                                         np.zeros_like(dz_active), route_ctx,
                                         e_lam, lam_id)
            e = measured_remainder(phi2, phi, dphi, dz_active)
            self.remainder.update(lam_id, e)
            if nxt.accepted_jvp % cfg.audit_period == 0:
                exact = exact_encode_fn()
                self.tel.audits += 1
                drift = float(np.linalg.norm(nxt.Z - exact.Z) /
                              (np.linalg.norm(exact.Z) + 1e-8))
                self.tel.audit_drift.append(drift)
                if drift > 0.5:
                    self.tel.false_stable += 1
                    return exact, False
                # re-anchor: replace approximate primal with the exact one
                return exact, True
            return nxt, True

        exact = exact_encode_fn()
        self.tel.exact_recompute += 1
        return exact, False
