from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx


class ArcDatasetClient:
    """Small synchronous client for training jobs and notebooks."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_seconds,
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "ArcDatasetClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def games(self) -> list[dict[str, Any]]:
        response = self.client.get("/v1/games")
        response.raise_for_status()
        return response.json()

    def recordings(
        self, game_code: str | None = None, page_size: int = 500
    ) -> Iterator[dict[str, Any]]:
        offset = 0
        while True:
            params: dict[str, Any] = {"limit": page_size, "offset": offset}
            if game_code:
                params["game_code"] = game_code
            response = self.client.get("/v1/recordings", params=params)
            response.raise_for_status()
            page = response.json()
            yield from page["items"]
            offset += len(page["items"])
            if offset >= page["total"] or not page["items"]:
                return

    def events(self, recording_id: str) -> Iterator[dict[str, Any]]:
        """Stream one recording line-by-line without loading it into memory."""
        with self.client.stream(
            "GET", f"/v1/recordings/{recording_id}/download"
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.strip():
                    yield json.loads(line)

    def download(self, recording_id: str, destination: str) -> None:
        with self.client.stream(
            "GET", f"/v1/recordings/{recording_id}/download"
        ) as response:
            response.raise_for_status()
            with open(destination, "wb") as output:
                for chunk in response.iter_bytes():
                    output.write(chunk)
