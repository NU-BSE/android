"""The lambda library Lambda = {lambda_1..lambda_K}, K = 128.

  lambda_0..lambda_4         : key operators ACTION1..ACTION5
  lambda_5..lambda_124       : CLICK(slot, offset) - 24 object slots x 5
                               offsets. Offset 0 clicks the object centroid;
                               offsets 1..4 click one object-size beside it
                               (up/down/left/right) - the objectness prior
                               "interact with a thing or right next to it".
  lambda_125..lambda_127     : reserved (always masked invalid)

Valid-action masking: key operators are valid iff the environment accepts
the key; click operators iff the slot indexes an existing object.

Each click lambda carries an *argument feature* vector: the target object's
token features plus an offset one-hot (ARG_DIM). The Q-router adds these to
the learned library embedding, giving state-conditioned, object-typed action
embeddings (shared action geometry across compare-like operators).
"""
from __future__ import annotations

import numpy as np

from ..env.core import ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, CLICK
from ..state import Obj, tokenize, TOKEN_DIM
from .base import OperatorMeta

N_KEYS = 5
N_SLOTS = 24
N_OFFSETS = 5           # center, up, down, left, right
K = 128
ARG_DIM = TOKEN_DIM + N_OFFSETS

_KEY_ACTIONS = [ACTION1, ACTION2, ACTION3, ACTION4, ACTION5]
_OFF = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

METAS: list[OperatorMeta] = []
for i in range(N_KEYS):
    METAS.append(OperatorMeta(i, arity=0, branch_count=2,
                              kind="move" if i < 4 else "key",
                              mutates_relations=True))
for lam in range(N_KEYS, K):
    METAS.append(OperatorMeta(lam, arity=1, branch_count=2, kind="click",
                              mutates_relations=True))

META_FEATURES = np.stack([m.features() for m in METAS])       # (K, 8)


def click_lambda(slot: int, offset: int) -> int:
    return N_KEYS + slot * N_OFFSETS + offset


def click_slot_offset(lam: int) -> tuple[int, int]:
    r = lam - N_KEYS
    return r // N_OFFSETS, r % N_OFFSETS


def valid_mask(env, objs: list[Obj]) -> np.ndarray:
    mask = np.zeros(K, dtype=bool)
    for i, a in enumerate(_KEY_ACTIONS):
        mask[i] = a in env.action_space
    if env.uses_click:
        n = min(len(objs), N_SLOTS)
        for s in range(n):
            mask[click_lambda(s, 0)] = True
            if objs[s].area < 200:      # offset clicks: only beside small things
                for o in range(1, N_OFFSETS):
                    mask[click_lambda(s, o)] = True
    return mask


def arg_features(objs: list[Obj]) -> np.ndarray:
    """(K, ARG_DIM) object-argument features per lambda (zeros for keys)."""
    F = np.zeros((K, ARG_DIM))
    feats = tokenize(objs) if objs else np.zeros((0, TOKEN_DIM))
    n = min(len(objs), N_SLOTS)
    for s in range(n):
        for o in range(N_OFFSETS):
            lam = click_lambda(s, o)
            F[lam, :TOKEN_DIM] = feats[s]
            F[lam, TOKEN_DIM + o] = 1.0
    return F


def to_env_action(lam: int, objs: list[Obj]):
    """Map a lambda to (action, x, y) for ArcEnv.step / the remote API."""
    if lam < N_KEYS:
        return _KEY_ACTIONS[lam], None, None
    slot, off = click_slot_offset(lam)
    obj = objs[slot]
    dy, dx = _OFF[off]
    h = max(3, obj.y1 - obj.y0 + 1)
    w = max(3, obj.x1 - obj.x0 + 1)
    x = int(np.clip(round(obj.cx + dx * w), 0, 63))
    y = int(np.clip(round(obj.cy + dy * h), 0, 63))
    return CLICK, x, y
