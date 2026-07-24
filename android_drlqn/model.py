"""Compact multimodal DRLQN for Android window trajectories.

The planner emits semantic subgoals. Coordinates are produced only by the
pointer/heatmap grounder and the gesture decoder.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .autodiff import T, LayerNorm, Linear, MLP, Module, Param
from .core import (
    ACCESS_NODE_DIM,
    ACTION_HISTORY_DIM,
    EVENT_DIM,
    META_DIM,
    TEXT_TOKEN_DIM,
)
from .window_graph import SubgoalType


class MultiHeadAttention(Module):
    def __init__(self, width: int, heads: int, rng: np.random.Generator):
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.q = Linear(width, width, rng)
        self.k = Linear(width, width, rng)
        self.v = Linear(width, width, rng)
        self.out = Linear(width, width, rng)

    def _split(self, x: T, batch: int, tokens: int) -> T:
        return x.reshape(batch, tokens, self.heads, self.head_width).transpose(0, 2, 1, 3)

    def __call__(self, query: T, context: T, key_mask: np.ndarray | None = None) -> T:
        batch, query_tokens = query.shape[:2]
        context_tokens = context.shape[1]
        q = self._split(self.q(query), batch, query_tokens)
        k = self._split(self.k(context), batch, context_tokens)
        v = self._split(self.v(context), batch, context_tokens)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (self.head_width ** -0.5)
        if key_mask is not None:
            scores = scores + T(np.where(key_mask, 0.0, -1e9)[:, None, None, :])
        weights = scores.softmax(axis=-1)
        mixed = (weights @ v).transpose(0, 2, 1, 3).reshape(
            batch, query_tokens, self.heads * self.head_width
        )
        return self.out(mixed)


class FusionBlock(Module):
    def __init__(self, width: int, heads: int, rng: np.random.Generator):
        self.attention = MultiHeadAttention(width, heads, rng)
        self.norm_attention = LayerNorm(width)
        self.feed_forward = MLP([width, 2 * width, width], rng)
        self.norm_feed_forward = LayerNorm(width)

    def __call__(self, tokens: T, key_mask: np.ndarray) -> T:
        tokens = self.norm_attention(tokens + self.attention(tokens, tokens, key_mask))
        return self.norm_feed_forward(tokens + self.feed_forward(tokens))


@dataclass
class AndroidDRLQNConfig:
    d_model: int = 128
    heads: int = 4
    fusion_blocks: int = 2
    history: int = 8
    max_nodes: int = 96
    heatmap_size: int = 16
    planner_args: int = 32
    gesture_types: int = 9
    max_fingers: int = 4
    control_points: int = 4
    verifier_labels: int = 6

    def __post_init__(self):
        if not 3 <= self.history <= 8:
            raise ValueError("history must be in [3, 8]")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")


class AndroidDRLQN(Module):
    """Small dependency-free model built on the repository's numpy autodiff."""

    def __init__(self, cfg: AndroidDRLQNConfig | None = None, seed: int = 0):
        cfg = cfg or AndroidDRLQNConfig()
        rng = np.random.default_rng(seed)
        width = cfg.d_model
        self.cfg = cfg

        self.vision_projection = Linear(3, width, rng)
        self.node_projection = Linear(ACCESS_NODE_DIM, width, rng)
        self.text_projection = Linear(TEXT_TOKEN_DIM, width, rng)
        self.event_projection = Linear(EVENT_DIM, width, rng)
        self.metadata_projection = Linear(META_DIM + ACTION_HISTORY_DIM, width, rng)
        self.modality_embeddings = Param(rng.normal(0, 0.15, size=(5, width)))

        self.fusion = [FusionBlock(width, cfg.heads, rng) for _ in range(cfg.fusion_blocks)]
        self.frame_query = Param(rng.normal(0, 0.15, size=(1, width)))
        self.frame_pool = MultiHeadAttention(width, cfg.heads, rng)

        # GRU-style temporal memory over the most recent 3-8 synchronized frames.
        self.temporal_input = Linear(width, 3 * width, rng)
        self.temporal_hidden = Linear(width, 3 * width, rng, bias=False)

        self.planner = MLP([width, width, len(SubgoalType)], rng)
        self.planner_argument = Linear(width, cfg.planner_args, rng)

        self.node_key = Linear(width, width, rng)
        self.node_query = Linear(width, width, rng)
        self.box_head = MLP([2 * width, width, 4], rng)
        self.heatmap_head = MLP([width, width, cfg.heatmap_size * cfg.heatmap_size], rng)

        gesture_parameters = cfg.max_fingers * cfg.control_points * 2
        self.gesture_type = Linear(width, cfg.gesture_types, rng)
        self.finger_count = Linear(width, cfg.max_fingers, rng)
        self.control_points = MLP([width, width, gesture_parameters], rng)
        self.duration = MLP([width, width, 1], rng)
        self.velocity = MLP([width, width, 3], rng)

        self.verifier = MLP([2 * width, width, cfg.verifier_labels], rng)
        self.world_model = MLP([2 * width, width, width], rng)

        # Differential residual lambda update: state + gated action-conditioned delta.
        self.lambda_gate = Linear(2 * width, width, rng)
        self.lambda_delta = MLP([2 * width, width, width], rng)

    def _vision_tokens(self, screenshots: np.ndarray) -> T:
        batch, history = screenshots.shape[:2]
        # 64x64 screenshots -> 8x8 grid of RGB patch means.
        patches = screenshots.reshape(batch, history, 8, 8, 8, 8, 3).mean(axis=(3, 5))
        return self.vision_projection(T(patches.reshape(batch * history, 64, 3)))

    def _encode_frames(self, batch: dict[str, np.ndarray]):
        width = self.cfg.d_model
        screenshots = batch["screenshots"]
        batch_size, history = screenshots.shape[:2]
        flat_batch = batch_size * history

        vision = self._vision_tokens(screenshots) + self.modality_embeddings[0].reshape(1, 1, width)
        nodes = self.node_projection(
            T(batch["node_features"].reshape(flat_batch, -1, ACCESS_NODE_DIM))
        ) + self.modality_embeddings[1].reshape(1, 1, width)
        texts = self.text_projection(
            T(batch["text_features"].reshape(flat_batch, -1, TEXT_TOKEN_DIM))
        ) + self.modality_embeddings[2].reshape(1, 1, width)
        events = self.event_projection(
            T(batch["events"].reshape(flat_batch, -1, EVENT_DIM))
        ) + self.modality_embeddings[3].reshape(1, 1, width)

        metadata = np.concatenate([batch["metadata"], batch["previous_actions"]], axis=-1)
        metadata = self.metadata_projection(T(metadata.reshape(flat_batch, 1, -1)))
        metadata = metadata + self.modality_embeddings[4].reshape(1, 1, width)

        tokens = T.concat([vision, nodes, texts, events, metadata], axis=1)
        valid = np.concatenate(
            [
                np.ones((flat_batch, vision.shape[1]), dtype=bool),
                batch["node_mask"].reshape(flat_batch, -1),
                batch["text_mask"].reshape(flat_batch, -1),
                batch["event_mask"].reshape(flat_batch, -1),
                np.ones((flat_batch, 1), dtype=bool),
            ],
            axis=1,
        )

        for block in self.fusion:
            tokens = block(tokens, valid)

        query = self.frame_query.reshape(1, 1, width) + T(np.zeros((flat_batch, 1, 1)))
        frames = self.frame_pool(query, tokens, valid).reshape(batch_size, history, width)
        latest_nodes = nodes.reshape(batch_size, history, -1, width)[:, -1]
        latest_mask = batch["node_mask"][:, -1]
        return frames, latest_nodes, latest_mask

    def _encode_history(self, frames: T, history_mask: np.ndarray) -> T:
        batch_size, history, width = frames.shape
        hidden = T(np.zeros((batch_size, width)))
        for step in range(history):
            gates = self.temporal_input(frames[:, step]) + self.temporal_hidden(hidden)
            update = gates[:, :width].sigmoid()
            reset = gates[:, width : 2 * width].sigmoid()
            candidate = (gates[:, 2 * width :] + reset * hidden).tanh()
            next_hidden = (1 - update) * hidden + update * candidate
            valid = T(history_mask[:, step : step + 1].astype(float))
            hidden = valid * next_hidden + (1 - valid) * hidden
        return hidden

    def lambda_update(self, state_latent: T, action_embedding: T) -> T:
        joined = T.concat([state_latent, action_embedding], axis=-1)
        gate = self.lambda_gate(joined).sigmoid()
        delta = self.lambda_delta(joined).tanh()
        return state_latent + gate * delta

    def __call__(self, batch: dict[str, np.ndarray], next_state_latent: T | None = None):
        frames, nodes, node_mask = self._encode_frames(batch)
        latent = self._encode_history(frames, batch["history_mask"])

        query = self.node_query(latent).reshape(latent.shape[0], 1, self.cfg.d_model)
        keys = self.node_key(nodes)
        node_logits = (keys * query).sum(axis=-1) * (self.cfg.d_model ** -0.5)
        node_logits = node_logits + T(np.where(node_mask, 0.0, -1e9))
        node_weights = node_logits.softmax(axis=-1).reshape(latent.shape[0], -1, 1)
        grounded = (nodes * node_weights).sum(axis=1)
        combined = T.concat([latent, grounded], axis=-1)

        outputs = {
            "state_latent": latent,
            "planner_logits": self.planner(latent),
            "planner_argument": self.planner_argument(latent),
            "ground_node_logits": node_logits,
            "ground_box": self.box_head(combined).sigmoid(),
            "ground_heatmap": self.heatmap_head(latent).reshape(
                latent.shape[0], self.cfg.heatmap_size, self.cfg.heatmap_size
            ),
            "gesture_type_logits": self.gesture_type(latent),
            "finger_count_logits": self.finger_count(latent),
            "control_points": self.control_points(latent).sigmoid().reshape(
                latent.shape[0], self.cfg.max_fingers, self.cfg.control_points, 2
            ),
            "duration_ms": self.duration(latent).exp(),
            "velocity_profile": self.velocity(latent),
            "next_state_prediction": self.world_model(combined),
        }
        if next_state_latent is not None:
            outputs["verifier_logits"] = self.verifier(
                T.concat([latent, next_state_latent], axis=-1)
            )
        return outputs
