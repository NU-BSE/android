"""Dependency-free client for the hosted ARC recording dataset."""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class DatasetProtocolError(RuntimeError):
    """Raised when the hosted API returns a successful but unsupported payload."""


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

    @staticmethod
    def _recording_items(payload: Any, *, endpoint: str) -> list[dict[str, Any]]:
        """Normalize page, manifest and legacy API response shapes.

        The checked-in FastAPI service returns ``{"items": [...]}`` for the page
        endpoint and ``{"recordings": [...]}`` for the manifest. Older deployed
        versions have also returned a bare list or wrapped either shape in ``data``.
        Treating an unknown successful payload as an empty list hid deployment
        incompatibilities, so unsupported shapes now fail explicitly.
        """
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, dict):
            values = None
            for key in ("items", "recordings"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    values = candidate
                    break
            if values is None and isinstance(payload.get("data"), (dict, list)):
                return ArcDatasetClient._recording_items(
                    payload["data"], endpoint=endpoint
                )
            if values is None:
                keys = ", ".join(sorted(map(str, payload.keys()))) or "<none>"
                raise DatasetProtocolError(
                    f"{endpoint} returned no recording list; response keys: {keys}"
                )
        else:
            raise DatasetProtocolError(
                f"{endpoint} returned {type(payload).__name__}, expected an object or list"
            )

        result = []
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                raise DatasetProtocolError(
                    f"{endpoint} recording {index} is {type(value).__name__}, expected object"
                )
            result.append(value)
        return result

    def stats(self) -> dict[str, Any]:
        payload = self._json("/v1/stats")
        if not isinstance(payload, dict):
            raise DatasetProtocolError("/v1/stats did not return a JSON object")
        return payload

    def manifest_recordings(self) -> list[dict[str, Any]]:
        payload = self._json("/v1/dataset/manifest")
        return self._recording_items(payload, endpoint="/v1/dataset/manifest")

    def get_recording(self, recording_id: str) -> dict[str, Any]:
        payload = self._json(f"/v1/recordings/{quote(recording_id, safe='')}")
        if not isinstance(payload, dict):
            raise DatasetProtocolError("recording metadata did not return a JSON object")
        return payload

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
            items = self._recording_items(page, endpoint="/v1/recordings")

            # Some older reverse-proxy deployments exposed the manifest correctly
            # while their paginated route returned an empty legacy envelope. Use the
            # manifest as a compatibility fallback, but only on the first empty page.
            if offset == 0 and not items:
                try:
                    manifest = self.manifest_recordings()
                except Exception:
                    manifest = []
                if game_code:
                    manifest = [
                        item for item in manifest
                        if str(item.get("game_code") or "").lower() == game_code.lower()
                    ]
                yield from manifest
                return

            yield from items
            offset += len(items)
            if not items:
                return

            total = None
            if isinstance(page, dict):
                total_raw = page.get("total")
                if isinstance(total_raw, (int, float)):
                    total = int(total_raw)
                elif isinstance(page.get("data"), dict):
                    nested_total = page["data"].get("total")
                    if isinstance(nested_total, (int, float)):
                        total = int(nested_total)
            if total is None or offset >= total:
                return

    def events(self, recording_id: str) -> Iterator[dict[str, Any]]:
        request = self._request(
            f"/v1/recordings/{quote(recording_id, safe='')}/download"
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            for line_number, raw_line in enumerate(response, start=1):
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetProtocolError(
                        f"recording {recording_id} has invalid JSON on line {line_number}: "
                        f"{exc.msg}"
                    ) from exc
                if not isinstance(value, dict):
                    raise DatasetProtocolError(
                        f"recording {recording_id} line {line_number} is not an object"
                    )
                yield value
