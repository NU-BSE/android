"""Policy-safe views of the Android DRLQN model.

The base research model contains grounding and gesture heads. AdvisoryDRLQN
keeps the original chat-only boundary. PerceptionDRLQN is an internal VLM
coordination facade that additionally exposes node/box/heatmap grounding while
still rejecting every gesture trajectory and timing output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .model import AndroidDRLQN, AndroidDRLQNConfig


GESTURE_OUTPUT_KEYS = frozenset({
    "gesture_type_logits",
    "finger_count_logits",
    "control_points",
    "duration_ms",
    "velocity_profile",
})

SPATIAL_OUTPUT_KEYS = frozenset({
    "ground_box",
    "ground_heatmap",
})

ACTIONABLE_OUTPUT_KEYS = GESTURE_OUTPUT_KEYS | SPATIAL_OUTPUT_KEYS

ADVISORY_OUTPUT_KEYS = frozenset({
    "state_latent",
    "planner_logits",
    "planner_argument",
    "ground_node_logits",
    "next_state_prediction",
    "verifier_logits",
})

PERCEPTION_OUTPUT_KEYS = ADVISORY_OUTPUT_KEYS | SPATIAL_OUTPUT_KEYS


@dataclass(frozen=True)
class AdvisoryModelConfig:
    expose_node_ranking: bool = True


class AdvisoryDRLQN:
    """Read-only planner/verifier facade with no executable spatial outputs."""

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


@dataclass(frozen=True)
class PerceptionModelConfig:
    expose_node_ranking: bool = True
    expose_box: bool = True
    expose_heatmap: bool = True


class PerceptionDRLQN:
    """Internal visual-grounding facade used by SettingsCoordinator.

    Spatial predictions can be consumed by a policy gate or fused with an
    external VLM, but gesture types, trajectories, durations and velocities can
    never cross this boundary.
    """

    def __init__(
        self,
        model: AndroidDRLQN | None = None,
        *,
        model_config: AndroidDRLQNConfig | None = None,
        config: PerceptionModelConfig | None = None,
        seed: int = 0,
    ):
        self.model = model or AndroidDRLQN(model_config, seed=seed)
        self.config = config or PerceptionModelConfig()

    def __call__(self, batch: Mapping[str, np.ndarray], next_state_latent=None) -> dict:
        raw = self.model(dict(batch), next_state_latent=next_state_latent)
        output = {key: value for key, value in raw.items() if key in PERCEPTION_OUTPUT_KEYS}
        if not self.config.expose_node_ranking:
            output.pop("ground_node_logits", None)
        if not self.config.expose_box:
            output.pop("ground_box", None)
        if not self.config.expose_heatmap:
            output.pop("ground_heatmap", None)
        self.assert_no_gesture_outputs(output)
        return output

    @staticmethod
    def assert_no_gesture_outputs(outputs: Mapping[str, object]) -> None:
        leaked = GESTURE_OUTPUT_KEYS.intersection(outputs)
        if leaked:
            raise RuntimeError(f"gesture outputs crossed perception boundary: {sorted(leaked)}")
