"""Policy-safe view of the Android DRLQN model.

The base research model contains grounding and gesture heads for offline
training. Production chat guidance must use :class:`AdvisoryDRLQN`, which
filters all coordinate, bounding-box and gesture outputs at the API boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .model import AndroidDRLQN, AndroidDRLQNConfig


ACTIONABLE_OUTPUT_KEYS = frozenset({
    "ground_box",
    "ground_heatmap",
    "gesture_type_logits",
    "finger_count_logits",
    "control_points",
    "duration_ms",
    "velocity_profile",
})

ADVISORY_OUTPUT_KEYS = frozenset({
    "state_latent",
    "planner_logits",
    "planner_argument",
    "ground_node_logits",
    "next_state_prediction",
    "verifier_logits",
})


@dataclass(frozen=True)
class AdvisoryModelConfig:
    expose_node_ranking: bool = True


class AdvisoryDRLQN:
    """Read-only planner/verifier facade with no executable gesture outputs."""

    def __init__(
        self,
        model: AndroidDRLQN | None = None,
        *,
        model_config: AndroidDRLQNConfig | None = None,
        config: AdvisoryModelConfig | None = None,
        seed: int = 0,
    ):
        self.model = model or AndroidDRLQN(model_config, seed=seed)
        self.config = config or AdvisoryModelConfig()

    def __call__(self, batch: Mapping[str, np.ndarray], next_state_latent=None) -> dict:
        raw = self.model(dict(batch), next_state_latent=next_state_latent)
        safe = {key: value for key, value in raw.items() if key in ADVISORY_OUTPUT_KEYS}
        if not self.config.expose_node_ranking:
            safe.pop("ground_node_logits", None)
        self.assert_advisory_only(safe)
        return safe

    @staticmethod
    def assert_advisory_only(outputs: Mapping[str, object]) -> None:
        leaked = ACTIONABLE_OUTPUT_KEYS.intersection(outputs)
        if leaked:
            raise RuntimeError(f"actionable DRLQN outputs crossed advisor boundary: {sorted(leaked)}")
