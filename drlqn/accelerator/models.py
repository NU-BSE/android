"""PyTorch student-policy candidates for architecture ablations.

This module is optional: the numpy submission runtime remains dependency-light.
Training machines may install PyTorch and export the best validated student or use
its results to decide whether the sparse executor/JVP path is worth retaining.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - optional accelerator dependency
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass(frozen=True)
class AcceleratorModelConfig:
    name: str
    actions: int = 128
    colors: int = 16
    width: int = 192
    depth: int = 4
    heads: int = 6
    recurrent: bool = False
    differential_runtime: bool = False


def require_torch():
    if torch is None or nn is None:
        raise RuntimeError("Install PyTorch on an accelerator-backed training host")
    return torch, nn


if nn is not None:  # pragma: no branch
    class _VisualBackbone(nn.Module):
        def __init__(self, colors: int, width: int):
            super().__init__()
            self.embedding = nn.Embedding(colors, 16)
            self.conv = nn.Sequential(
                nn.Conv2d(16, width // 2, 5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv2d(width // 2, width, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(width, width, 3, stride=2, padding=1),
                nn.GELU(),
            )

        def forward(self, grid):
            x = self.embedding(grid.long()).permute(0, 3, 1, 2)
            return self.conv(x).mean(dim=(-2, -1))


    class LargeDRLQNPolicy(nn.Module):
        """Larger action-embedding router with an optional residual executor head."""

        def __init__(self, config: AcceleratorModelConfig):
            super().__init__()
            self.config = config
            self.backbone = _VisualBackbone(config.colors, config.width)
            self.state = nn.Sequential(
                nn.LayerNorm(config.width),
                nn.Linear(config.width, config.width),
                nn.GELU(),
                nn.Linear(config.width, config.width),
            )
            self.action_embedding = nn.Parameter(
                torch.randn(config.actions, config.width) * 0.02
            )
            self.value = nn.Linear(config.width, 1)
            self.frame_change = nn.Linear(config.width, 1)
            self.level_advance = nn.Linear(config.width, 1)
            if config.differential_runtime:
                self.executor = nn.Sequential(
                    nn.Linear(config.width * 2, config.width * 2),
                    nn.GELU(),
                    nn.Linear(config.width * 2, config.width),
                )
                self.next_embedding = None
            else:
                self.executor = None
                self.next_embedding = nn.Linear(config.width, config.width)

        def encode(self, grid):
            return self.state(self.backbone(grid))

        def _predict_next(self, state, logits, selected_action=None):
            if self.executor is None:
                return self.next_embedding(state)
            if selected_action is None:
                selected_action = logits.argmax(dim=-1)
            action = self.action_embedding[selected_action.long()]
            return state + self.executor(torch.cat([state, action], dim=-1))

        def forward(
            self,
            grid,
            previous_effect=None,
            selected_action=None,
            hidden=None,
        ):
            del previous_effect, hidden
            state = self.encode(grid)
            advantage = state @ self.action_embedding.t() / (state.shape[-1] ** 0.5)
            logits = self.value(state) + advantage - advantage.mean(dim=-1, keepdim=True)
            return {
                "logits": logits,
                "embedding": state,
                "frame_change": self.frame_change(state).squeeze(-1),
                "level_advance": self.level_advance(state).squeeze(-1),
                "next_embedding": self._predict_next(state, logits, selected_action),
            }


    class RecurrentVisualPolicy(nn.Module):
        """Visual encoder plus bounded recurrent memory over recent action effects."""

        def __init__(self, config: AcceleratorModelConfig):
            super().__init__()
            self.config = config
            self.backbone = _VisualBackbone(config.colors, config.width)
            self.effect = nn.Sequential(nn.Linear(6, config.width), nn.GELU())
            self.gru = nn.GRUCell(config.width * 2, config.width)
            self.router = nn.Linear(config.width, config.actions)
            self.frame_change = nn.Linear(config.width, 1)
            self.level_advance = nn.Linear(config.width, 1)
            self.next_embedding = nn.Linear(config.width, config.width)

        def encode(self, grid, previous_effect=None, hidden=None):
            visual = self.backbone(grid)
            if previous_effect is None:
                previous_effect = visual.new_zeros((visual.shape[0], 6))
            effect = self.effect(previous_effect.float())
            if hidden is None:
                hidden = visual.new_zeros((visual.shape[0], self.config.width))
            return self.gru(torch.cat([visual, effect], dim=-1), hidden)

        def forward(
            self,
            grid,
            previous_effect=None,
            selected_action=None,
            hidden=None,
        ):
            del selected_action
            state = self.encode(grid, previous_effect, hidden)
            return {
                "logits": self.router(state),
                "embedding": state,
                "hidden": state,
                "frame_change": self.frame_change(state).squeeze(-1),
                "level_advance": self.level_advance(state).squeeze(-1),
                "next_embedding": self.next_embedding(state),
            }


    class ObjectTileTransformerPolicy(nn.Module):
        """Transformer over object summaries, tile patches and action-effect tokens."""

        def __init__(self, config: AcceleratorModelConfig):
            super().__init__()
            self.config = config
            patch = 8
            self.color = nn.Embedding(config.colors, 16)
            self.patch = nn.Conv2d(16, config.width, patch, stride=patch)
            self.object_color = nn.Embedding(config.colors, config.width)
            self.object_features = nn.Linear(3, config.width)
            self.effect = nn.Linear(6, config.width)
            self.cls = nn.Parameter(torch.zeros(1, 1, config.width))
            token_count = 1 + 1 + config.colors + 64
            self.position = nn.Parameter(
                torch.randn(1, token_count, config.width) * 0.01
            )
            layer = nn.TransformerEncoderLayer(
                d_model=config.width,
                nhead=config.heads,
                dim_feedforward=config.width * 4,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.depth)
            self.router = nn.Linear(config.width, config.actions)
            self.frame_change = nn.Linear(config.width, 1)
            self.level_advance = nn.Linear(config.width, 1)
            self.next_embedding = nn.Linear(config.width, config.width)

        def _object_tokens(self, grid):
            one_hot = torch.nn.functional.one_hot(
                grid.long(), num_classes=self.config.colors
            ).float()
            mass = one_hot.mean(dim=(1, 2))
            height, width = grid.shape[-2:]
            ys = torch.linspace(0.0, 1.0, height, device=grid.device).view(1, height, 1, 1)
            xs = torch.linspace(0.0, 1.0, width, device=grid.device).view(1, 1, width, 1)
            normalizer = one_hot.sum(dim=(1, 2)).clamp_min(1.0)
            cy = (one_hot * ys).sum(dim=(1, 2)) / normalizer
            cx = (one_hot * xs).sum(dim=(1, 2)) / normalizer
            features = torch.stack([mass, cy, cx], dim=-1)
            colors = torch.arange(self.config.colors, device=grid.device)
            return self.object_features(features) + self.object_color(colors).unsqueeze(0)

        def encode(self, grid, previous_effect=None):
            x = self.color(grid.long()).permute(0, 3, 1, 2)
            patches = self.patch(x).flatten(2).transpose(1, 2)
            objects = self._object_tokens(grid)
            if previous_effect is None:
                previous_effect = patches.new_zeros((patches.shape[0], 6))
            effect = self.effect(previous_effect.float()).unsqueeze(1)
            cls = self.cls.expand(patches.shape[0], -1, -1)
            tokens = torch.cat([cls, effect, objects, patches], dim=1)
            tokens = tokens + self.position[:, : tokens.shape[1]]
            return self.encoder(tokens)[:, 0]

        def forward(
            self,
            grid,
            previous_effect=None,
            selected_action=None,
            hidden=None,
        ):
            del selected_action, hidden
            state = self.encode(grid, previous_effect)
            return {
                "logits": self.router(state),
                "embedding": state,
                "frame_change": self.frame_change(state).squeeze(-1),
                "level_advance": self.level_advance(state).squeeze(-1),
                "next_embedding": self.next_embedding(state),
            }


def build_accelerator_model(name: str, **overrides: Any):
    require_torch()
    normalized = name.strip().lower()
    configs = {
        "large-drlqn": AcceleratorModelConfig(
            "large-drlqn", width=256, depth=6, differential_runtime=True
        ),
        "recurrent-visual": AcceleratorModelConfig(
            "recurrent-visual", width=192, recurrent=True
        ),
        "tile-transformer": AcceleratorModelConfig(
            "tile-transformer", width=192, depth=4, heads=6
        ),
        "drlqn-no-jvp": AcceleratorModelConfig(
            "drlqn-no-jvp", width=256, depth=6, differential_runtime=False
        ),
    }
    if normalized not in configs:
        raise ValueError(f"unknown accelerator model: {name}")
    config = configs[normalized]
    if overrides:
        config = AcceleratorModelConfig(**{**config.__dict__, **overrides})
    if normalized == "recurrent-visual":
        return RecurrentVisualPolicy(config)
    if normalized == "tile-transformer":
        return ObjectTileTransformerPolicy(config)
    return LargeDRLQNPolicy(config)
