"""DRLQN solver: online (first-contact) training + hard tree routing.

The tree/graph over environment states is the authoritative structure
(hard semantics). The Q-router chooses which lambda to expand at a node;
frontier routing follows argmin path cost over the *known* deterministic
graph (the practical form of "hard route: argmin path violation" - a node
with untried lambdas has non-zero information value). The differential
shadow (JVP world-model updates guarded by stability tests) replaces exact
encoder recomputation where provably safe (phase P5).

Curriculum P0..P5 (spec 7.5) is realised inside one first-contact run:
  P0  operator/env verification lives in tests/
  P1  Q baseline, exact execution           (updates < diff_losses_at)
  P2  sparse routing is structurally always on
  P3  + differential losses L_dQ, L_op      (updates >= diff_losses_at)
  P4  shadow metrics recorded, no serving
  P5  approximate recompute with guards     (updates >= approx_at)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..autodiff import T, Adam
from ..config import DRLQNConfig
from ..model.drlqn import DRLQN
from ..operators.library import K, ARG_DIM, valid_mask, to_env_action, arg_features, click_slot_offset
from ..runtime.cache import RuntimeTelemetry
from ..runtime.recompute import DifferentialRuntime
from ..runtime.shadow import shadow_report
from ..state import extract_objects, tokenize, frame_key, TraceSummary, MAX_TOKENS, TOKEN_DIM
from .replay import Replay, Transition
from .losses import (double_dqn_targets, q_loss, world_model_losses,
                     differential_q_loss, operator_linearity_loss)


@dataclass
class PhaseSchedule:
    diff_losses_at: int = 100      # P3
    approx_at: int = 200           # P5
    target_sync: int = 100
    train_every: int = 2
    batch: int = 8
    warmup: int = 24               # replay size before updates start


@dataclass
class NodeInfo:
    key: str
    tokens: np.ndarray
    token_mask: np.ndarray
    trace: np.ndarray
    objs: list
    valid: np.ndarray
    level: int
    lam_feats: np.ndarray      # (K, ARG_DIM) object-argument features per lambda
    grid: np.ndarray           # raw frame (for semantic deltas)


@dataclass
class SolveResult:
    env_id: str
    completed: int = 0
    won: bool = False
    actions_per_level: dict = field(default_factory=dict)
    total_actions: int = 0
    updates: int = 0
    telemetry: dict = field(default_factory=dict)
    wall_clock_s: float = 0.0
    column_usage: list = field(default_factory=list)
    q_margin_mean: float = 0.0
    delta_support_mean: float = 0.0


class DRLQNSolver:
    def __init__(self, cfg: DRLQNConfig, seed: int = 0,
                 schedule: PhaseSchedule | None = None,
                 gamma: float = 0.9, lr: float = 3e-3,
                 mu_q: float = 1e-3, mu_op: float = 1e-4, mu_r: float = 1e-4):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.model = DRLQN(cfg, seed=seed)
        self.target = DRLQN(cfg, seed=seed)
        self.target.load_state_dict(self.model.state_dict())
        self.opt = Adam(self.model.parameters(), lr=lr)
        self.replay = Replay(seed=seed)
        self.sched = schedule or PhaseSchedule()
        self.gamma, self.mu_q, self.mu_op, self.mu_r = gamma, mu_q, mu_op, mu_r
        self.updates = 0
        self.nodes: dict[str, NodeInfo] = {}
        self.graph: dict[str, dict[int, str]] = {}
        self.caches: dict[str, tuple[int, object]] = {}
        self.runtime = DifferentialRuntime(self.model, cfg)
        self.column_usage = np.zeros(cfg.C)
        self.q_margins: list[float] = []
        self.delta_supports: list[float] = []
        # --- core-knowledge exploration priors (agentness / affordance) ---
        self.afford: dict[tuple, list[int]] = {}   # (color,off,size) -> [tries, changes]
        self.self_votes: dict[int, int] = {}       # color -> votes it is "me"
        self.move_dir: dict[int, np.ndarray] = {}  # key lambda -> mean (dy,dx)
        self.last_change: tuple | None = None      # centroid of last frame delta

    # -------------------------------------------------- exploration priors
    @staticmethod
    def _size_bucket(area: int) -> int:
        return 0 if area < 20 else (1 if area < 200 else 2)

    def _afford_key(self, node: NodeInfo, lam: int):
        if lam < 5:
            return ("key", lam, 0)
        slot, off = click_slot_offset(lam)
        if slot >= len(node.objs):
            return None
        o = node.objs[slot]
        return (o.color, off, self._size_bucket(o.area))

    def _update_priors(self, cur: NodeInfo, lam: int, nxt: NodeInfo, changed: bool):
        k = self._afford_key(cur, lam)
        if k is not None:
            t = self.afford.setdefault(k, [0, 0])
            t[0] += 1
            t[1] += int(changed)
        if changed and cur.level == nxt.level:
            ys, xs = np.where(cur.grid != nxt.grid)
            if len(ys):
                self.last_change = (float(ys.mean()), float(xs.mean()))
        elif cur.level != nxt.level:
            self.last_change = None
        if lam < 4 and changed:                    # possible self-object motion
            cur_by = {}
            for o in cur.objs:
                cur_by.setdefault(o.color, []).append(o)
            for o2 in nxt.objs:
                group = cur_by.get(o2.color, [])
                if len(group) != 1 or sum(1 for o in nxt.objs if o.color == o2.color) != 1:
                    continue
                o1 = group[0]
                if abs(o1.area - o2.area) > 0.2 * max(o1.area, 1):
                    continue
                d = np.array([o2.cy - o1.cy, o2.cx - o1.cx])
                if 0.5 < np.abs(d).sum() < 16:
                    self.self_votes[o2.color] = self.self_votes.get(o2.color, 0) + 1
                    if self.self_votes[o2.color] >= 2 and \
                            self.self_votes[o2.color] == max(self.self_votes.values()):
                        acc = self.move_dir.setdefault(lam, np.zeros(2))
                        self.move_dir[lam] = 0.7 * acc + 0.3 * d

    def _self_color(self) -> int | None:
        if not self.self_votes:
            return None
        c, v = max(self.self_votes.items(), key=lambda kv: kv[1])
        return c if v >= 2 else None

    def _prior_bonus(self, node: NodeInfo, lam: int) -> float:
        bonus = 0.0
        k = self._afford_key(node, lam)
        if k is not None and k[0] != "key":
            t = self.afford.get(k, [0, 0])
            bonus += 3.0 * (t[1] + 1) / (t[0] + 2)          # affordance ratio
            if k[1] == 0:
                bonus += 0.3                                # centre-click prior
            _, cx, cy = to_env_action(lam, node.objs)
            specials = [o for o in node.objs
                        if o.area < 400 and
                        sum(1 for p in node.objs if p.color == o.color) == 1]
            if specials:
                near = min(abs(o.cy - cy) + abs(o.cx - cx) for o in specials)
                bonus += 0.8 * max(0.0, 1.0 - near / 24.0)  # act near salient objects
                far_special = max(specials,
                                  key=lambda o: abs(o.cy - cy) + abs(o.cx - cx))
                d_far = abs(far_special.cy - cy) + abs(far_special.cx - cx)
                bonus += 1.0 * max(0.0, 1.0 - d_far / 96.0)  # progress toward the other
            if self.last_change is not None:
                d = abs(self.last_change[0] - cy) + abs(self.last_change[1] - cx)
                bonus += 1.5 * max(0.0, 1.0 - d / 16.0)      # continue where it changed
        if lam < 4:
            sc = self._self_color()
            d = self.move_dir.get(lam)
            if sc is not None and d is not None and np.abs(d).sum() > 0.3:
                me = next((o for o in node.objs if o.color == sc), None)
                if me is not None:
                    targets = [o for o in node.objs
                               if o.color != sc and o.area < 400
                               and sum(1 for p in node.objs if p.color == o.color) <= 2]
                    if targets:
                        tgt = min(targets, key=lambda o: abs(o.cy - me.cy) + abs(o.cx - me.cx))
                        v = np.array([tgt.cy - me.cy, tgt.cx - me.cx])
                        nv, nd = np.linalg.norm(v), np.linalg.norm(d)
                        if nv > 1e-6 and nd > 1e-6:
                            bonus += 0.8 * float(v @ d) / (nv * nd)
        return bonus

    # ------------------------------------------------------------- perception
    def perceive(self, env, grid) -> NodeInfo:
        key = frame_key(grid, env.level)
        if key in self.nodes:
            return self.nodes[key]
        objs = extract_objects(grid)
        feats = tokenize(objs)
        n = feats.shape[0]
        tokens = np.zeros((MAX_TOKENS, TOKEN_DIM))
        tokens[:n] = feats
        mask = np.zeros(MAX_TOKENS, dtype=bool)
        mask[:n] = True
        trace = TraceSummary(steps_in_level=0).features()
        lam_feats = arg_features(objs)
        node = NodeInfo(key, tokens, mask, trace, objs,
                        valid_mask(env, objs), env.level, lam_feats, grid.copy())
        self.nodes[key] = node
        return node

    # ------------------------------------------------------------ latent cache
    def _cache_version(self) -> int:
        return self.updates // 50

    def latent(self, node: NodeInfo):
        v = self._cache_version()
        hit = self.caches.get(node.key)
        if hit is not None and hit[0] == v:
            return hit[1]
        nc = self.runtime.exact_encode(node.tokens, node.token_mask, node.trace)
        self.caches[node.key] = (v, nc)
        return nc

    # ---------------------------------------------------------- action choice
    def q_of(self, node: NodeInfo) -> np.ndarray:
        nc = self.latent(node)
        Z = T(nc.Z[None])
        rc = T(np.zeros((1, self.cfg.d_route)))
        q = self.model.q_values(Z, rc, node.valid[None],
                                node.lam_feats[None]).data[0]
        vals = np.sort(q[node.valid])[::-1]
        if len(vals) > 1:
            self.q_margins.append(float(vals[0] - vals[1]))
        return q

    def select(self, node: NodeInfo, actions_taken: int) -> int | None:
        tried = self.graph.get(node.key, {})
        valid_ids = np.where(node.valid)[0]
        # dedupe click lambdas that resolve to coordinates already tried here
        tried_coords = {to_env_action(l, node.objs)[1:] for l in tried if l >= 5}
        untried = []
        seen_coords = set(tried_coords)
        for l in valid_ids:
            if l in tried:
                continue
            if l >= 5:
                xy = to_env_action(l, node.objs)[1:]
                if xy in seen_coords:
                    continue
                seen_coords.add(xy)
            untried.append(int(l))
        eps = max(0.08, 0.35 * np.exp(-actions_taken / 400.0))
        if untried:
            if self.rng.random() < eps:
                return int(self.rng.choice(untried))
            q = self.q_of(node)
            return int(max(untried, key=lambda l: q[l] + self._prior_bonus(node, l)))
        # frontier routing: BFS through the known deterministic graph
        step = self._route_to_frontier(node)
        return step

    def _route_to_frontier(self, node: NodeInfo) -> int | None:
        q = deque([node.key])
        first = {node.key: None}
        while q:
            k = q.popleft()
            info = self.nodes[k]
            tried = self.graph.get(k, {})
            untried = [l for l in np.where(info.valid)[0] if l not in tried]
            if untried and k != node.key:
                # walk back to the first edge out of `node`
                lam = first[k]
                return lam
            for lam, nk in tried.items():
                if nk in first or nk == k:
                    continue
                if self.nodes[nk].level != node.level:
                    continue
                first[nk] = first[k] if first[k] is not None else lam
                q.append(nk)
        return None

    # -------------------------------------------------------------- main loop
    def run_env(self, env, budgets: dict[int, int]) -> SolveResult:
        t0 = time.time()
        res = SolveResult(env_id=env.env_id)
        frame = env.reset()
        cur = self.perceive(env, frame.grid)
        prev_dz = None
        last_level = env.level
        while env.state == "NOT_FINISHED":
            lvl = env.level
            used = env._level_action_counts.get(lvl, 0)
            if used >= budgets.get(lvl, 10 ** 9):
                break                                    # budget exhausted -> fail here
            lam = self.select(cur, used)
            if lam is None:                              # exhausted reachable graph
                frame = env.reset_level()
                cur = self.perceive(env, frame.grid)
                prev_dz = None
                continue
            a, x, y = to_env_action(lam, cur.objs)
            step = env.step(a, x, y)
            nxt = self.perceive(env, step.frame.grid)
            changed = nxt.key != cur.key
            is_new = nxt.key not in self.graph and changed and not step.level_won
            reward = (-0.02
                      + (20.0 if step.env_won else 0.0)
                      + (10.0 if step.level_won and not step.env_won else 0.0)
                      + (0.5 if is_new else 0.0)
                      + (-0.3 if not changed else 0.0))
            self.graph.setdefault(cur.key, {})[lam] = nxt.key
            self._update_priors(cur, lam, nxt, changed)
            self.replay.push(Transition(
                cur.key, lam, reward, nxt.key, step.env_won,
                stable_pair=changed and not step.level_won, changed=changed))
            # ------------------------- differential runtime for the next latent
            if step.level_won:
                prev_dz = None                            # re-anchor at new level
            elif self.updates >= self.sched.approx_at and changed:
                cur_nc = self.latent(cur)
                dz_in = prev_dz if prev_dz is not None else np.zeros_like(cur_nc.Z)
                nxt_node = nxt

                def exact_fn():
                    nc = self.runtime.exact_encode(nxt_node.tokens,
                                                   nxt_node.token_mask,
                                                   nxt_node.trace)
                    self.caches[nxt_node.key] = (self._cache_version(), nc)
                    return nc

                e_lam = self.model.q_router.effective_embedding_np(
                    lam, cur.lam_feats[lam])
                nc2, approx = self.runtime.apply_delta(cur_nc, lam, dz_in,
                                                       np.zeros(self.cfg.d_route),
                                                       e_lam, exact_fn)
                self.caches[nxt.key] = (self._cache_version(), nc2)
                prev_dz = (nc2.Z - cur_nc.Z) if approx else None
                if nc2.margin is not None:
                    rep = shadow_report(nc2.margin, np.zeros_like(nc2.margin))
                    nc2.shadow_weights = rep[0]["p"]
                if nc2.active_indices is not None:
                    self.column_usage[nc2.active_indices] += 1
                dsup = float(np.mean(np.abs(nc2.Z - cur_nc.Z).sum(-1) > 1e-6))
                self.delta_supports.append(dsup)
            else:
                prev_dz = None
            # ------------------------------------------------------- learning
            if (len(self.replay) >= self.sched.warmup
                    and env.action_count % self.sched.train_every == 0):
                self.update()
            cur = nxt
            if step.level_won:
                res.completed = lvl
            if step.env_won:
                res.won = True
                break
            last_level = env.level
        res.actions_per_level = env.actions_per_level
        res.total_actions = env.action_count
        res.updates = self.updates
        res.telemetry = self.runtime.tel.summary()
        res.wall_clock_s = time.time() - t0
        res.column_usage = self.column_usage.tolist()
        res.q_margin_mean = float(np.mean(self.q_margins)) if self.q_margins else 0.0
        res.delta_support_mean = (float(np.mean(self.delta_supports))
                                  if self.delta_supports else 0.0)
        return res

    # ---------------------------------------------------------------- update
    def update(self):
        sch = self.sched
        batch = self.replay.sample(sch.batch)
        B = len(batch)
        cfg = self.cfg

        def gather(keys):
            toks = np.stack([self.nodes[k].tokens for k in keys])
            msk = np.stack([self.nodes[k].token_mask for k in keys])
            trc = np.stack([self.nodes[k].trace for k in keys])
            val = np.stack([self.nodes[k].valid for k in keys])
            of = np.stack([self.nodes[k].lam_feats for k in keys])
            return toks, msk, trc, val, of

        s_keys = [tr.key for tr in batch]
        n_keys = [tr.next_key for tr in batch]
        toks, msk, trc, valid, ofeat = gather(s_keys)
        ntoks, nmsk, ntrc, nvalid, nofeat = gather(n_keys)
        lams = np.array([tr.lam for tr in batch])
        rewards = np.array([tr.reward for tr in batch])
        dones = np.array([1.0 if tr.done else 0.0 for tr in batch])
        changed = np.array([tr.changed for tr in batch])
        stable = np.array([tr.stable_pair for tr in batch])

        targets = double_dqn_targets(self.model, self.target,
                                     (ntoks, nmsk, ntrc), nvalid,
                                     rewards, dones, self.gamma, nofeat)
        rc = T(np.zeros((B, cfg.d_route)))
        Z = self.model.encode(toks, msk, trc)
        Zn = self.model.encode(ntoks, nmsk, ntrc)          # engine (for L_dQ)
        Zn_sg = Zn.data.copy()

        lam_feats = ofeat[np.arange(B), lams]              # (B, ARG_DIM)
        total = q_loss(self.model, Z, rc, valid, lams, targets, ofeat)
        l_model, l_margin, l_router, margins, gap = world_model_losses(
            self.model, Z, rc, lams, lam_feats, Zn_sg, changed, stable)
        total = total + 0.5 * l_model + 0.2 * l_margin + self.mu_r * l_router

        if self.updates >= sch.diff_losses_at:
            idx = [i for i in range(B) if stable[i]][:2]
            if idx:
                nodes_Z = [Z.data[i] for i in idx]
                dZs = [Zn_sg[i] - Z.data[i] for i in idx]
                pairs = [(Z[i:i + 1], Zn[i:i + 1]) for i in idx]
                e_lams = [self.model.q_router.effective_embedding_np(
                    int(lams[i]), lam_feats[i]) for i in idx]
                l_dq = differential_q_loss(self.model, nodes_Z, dZs,
                                           np.zeros(cfg.d_route),
                                           [int(lams[i]) for i in idx],
                                           e_lams, [lam_feats[i] for i in idx],
                                           pairs)
                total = total + self.mu_q * l_dq
                i = idx[0]
                e_np = e_lams[0]
                act = np.where(
                    self._topk_mask_for(Z.data[i], e_np))[0]
                z_act = Z[i:i + 1][:, act]
                dz = 0.5 * (Zn_sg[i] - Z.data[i])[act]
                e_lam_T = (self.model.q_router.base_embeddings()[int(lams[i])]
                           + self.model.q_router.obj_proj(T(lam_feats[i]))
                           ).reshape(1, cfg.d_A)
                l_op = operator_linearity_loss(self.model, z_act, dz,
                                               np.zeros(cfg.d_route), e_lam_T,
                                               int(lams[i]))
                total = total + self.mu_op * l_op

        self.opt.zero_grad()
        total.backward()
        self.opt.step()
        self.updates += 1
        if self.updates % sch.target_sync == 0:
            self.target.load_state_dict(self.model.state_dict())

    def _topk_mask_for(self, Z: np.ndarray, e: np.ndarray) -> np.ndarray:
        from ..runtime.jvp import jvp_controller, jvp_column_scores
        I, _ = jvp_controller(self.model.controller, Z, np.zeros_like(Z),
                              np.zeros(self.cfg.d_route))
        rho, _ = jvp_column_scores(self.model.column_router, Z,
                                   np.zeros_like(Z), I, np.zeros_like(I), e)
        mask = np.zeros(self.cfg.C, dtype=bool)
        mask[np.argsort(-rho)[:self.cfg.top_k]] = True
        return mask
