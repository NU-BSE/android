"""Dependency-free client for the hosted ARC recording dataset."""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ArcDatasetClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)

    def _request(self, path: str, query: dict[str, Any] | None = None) -> Request:
        url = f"{self.base_url}{path}"
        if query:
            url += "?" + urlencode(query)
        headers = {"Accept": "application/json, application/x-ndjson"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return Request(url, headers=headers)

    def _json(self, path: str, query: dict[str, Any] | None = None) -> Any:
        request = self._request(path, query)
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.load(response)

    def get_recording(self, recording_id: str) -> dict[str, Any]:
        return self._json(f"/v1/recordings/{recording_id}")

    def recordings(
        self,
        game_code: str | None = None,
        page_size: int = 500,
    ) -> Iterator[dict[str, Any]]:
        offset = 0
        while True:
            query: dict[str, Any] = {"limit": int(page_size), "offset": offset}
            if game_code:
                query["game_code"] = game_code
            page = self._json("/v1/recordings", query)
            items = list(page.get("items") or [])
            yield from items
            offset += len(items)
            if not items or offset >= int(page.get("total", offset)):
                return

    def events(self, recording_id: str) -> Iterator[dict[str, Any]]:
        request = self._request(f"/v1/recordings/{recording_id}/download")
        with urlopen(request, timeout=self.timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if line:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield value
