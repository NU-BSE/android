"""Client for the real ARC-AGI-3 API (three.arcprize.org).

Requires an ARC API key (env var ARC_API_KEY). Not available in this
sandbox - the endpoint answers 401 without it - but this adapter lets the
identical DRLQN agent run against the official environments.

API surface (per the public agents repo):
  GET  /api/games                          -> [{game_id, title}, ...]
  POST /api/scorecard/open                 -> {card_id}
  POST /api/cmd/RESET  {game_id, card_id}  -> frame
  POST /api/cmd/ACTION1..ACTION5 {game_id, guid} -> frame
  POST /api/cmd/ACTION6 {game_id, guid, x, y}    -> frame (click)
  POST /api/scorecard/close {card_id}
"""
from __future__ import annotations

import os

import numpy as np
import requests

BASE = "https://three.arcprize.org"


class ArcClient:
    def __init__(self, api_key: str | None = None):
        self.key = api_key or os.environ.get("ARC_API_KEY")
        if not self.key:
            raise RuntimeError("ARC_API_KEY is required to reach three.arcprize.org")
        self.s = requests.Session()
        self.s.headers.update({"X-API-Key": self.key,
                               "Accept": "application/json"})

    def _post(self, path: str, payload: dict) -> dict:
        r = self.s.post(f"{BASE}{path}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def list_games(self) -> list[dict]:
        r = self.s.get(f"{BASE}/api/games", timeout=60)
        r.raise_for_status()
        return r.json()

    def open_scorecard(self, tags: list[str] | None = None) -> str:
        return self._post("/api/scorecard/open", {"tags": tags or ["drlqn"]})["card_id"]

    def close_scorecard(self, card_id: str) -> dict:
        return self._post("/api/scorecard/close", {"card_id": card_id})

    def reset(self, game_id: str, card_id: str) -> dict:
        return self._post("/api/cmd/RESET", {"game_id": game_id, "card_id": card_id})

    def act(self, game_id: str, guid: str, action: int,
            x: int | None = None, y: int | None = None) -> dict:
        name = f"ACTION{action}" if action <= 5 else "ACTION6"
        payload = {"game_id": game_id, "guid": guid}
        if action == 6:
            payload.update({"x": int(x), "y": int(y)})
        return self._post(f"/api/cmd/{name}", payload)


class RemoteEnv:
    """Adapts the remote API to the local ArcEnv interface used by the solver."""

    uses_click = True

    def __init__(self, client: ArcClient, game_id: str, card_id: str):
        self.client = client
        self.env_id = game_id[:4]
        self.game_id = game_id
        self.card_id = card_id
        self.guid = None
        self.level = 1
        self.state = "NOT_PLAYED"
        self.action_count = 0
        self._level_action_counts: dict[int, int] = {}
        self._grid = None
        # remote envs advertise their action space per frame; start permissive
        self.action_space = (1, 2, 3, 4, 5)

    def _ingest(self, data: dict):
        frames = data.get("frame") or []
        grid = np.array(frames[-1], dtype=np.uint8) if frames else self._grid
        self._grid = grid
        self.guid = data.get("guid", self.guid)
        self.state = data.get("state", self.state)
        new_level = int(data.get("score", 0)) + 1     # score == levels completed
        won_level = new_level > self.level
        self.level = new_level
        return grid, won_level

    def reset(self):
        data = self.client.reset(self.game_id, self.card_id)
        grid, _ = self._ingest(data)
        self.state = "NOT_FINISHED" if self.state != "WIN" else self.state
        from .agent_types import SimpleFrame
        return SimpleFrame(grid, self.state, self.level, self.action_count)

    def reset_level(self):
        self._count(self.level)
        return self.reset()

    def step(self, action: int, x: int | None = None, y: int | None = None):
        self._count(self.level)
        data = self.client.act(self.game_id, self.guid, action, x, y)
        grid, won_level = self._ingest(data)
        from .agent_types import SimpleFrame, SimpleStep
        f = SimpleFrame(grid, self.state, self.level, self.action_count)
        return SimpleStep(f, level_won=won_level, env_won=self.state == "WIN")

    def _count(self, lvl):
        self.action_count += 1
        self._level_action_counts[lvl] = self._level_action_counts.get(lvl, 0) + 1

    @property
    def actions_per_level(self):
        return dict(self._level_action_counts)
