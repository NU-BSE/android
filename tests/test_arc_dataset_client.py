from __future__ import annotations

from collections import defaultdict

import pytest

from drlqn.data.arc_dataset import ArcDatasetClient, DatasetProtocolError


class StubClient(ArcDatasetClient):
    def __init__(self, responses):
        super().__init__("https://example.invalid/agi-3", api_key="secret")
        self.responses = defaultdict(list)
        for path, values in responses.items():
            self.responses[path].extend(values)
        self.calls = []

    def _json(self, path, query=None):
        self.calls.append((path, query))
        values = self.responses[path]
        if not values:
            raise AssertionError(f"unexpected request: {path} {query}")
        return values.pop(0)


def test_recordings_accepts_legacy_recordings_envelope():
    client = StubClient({
        "/v1/recordings": [
            {"recordings": [{"id": "r1", "game_code": "aa00"}], "total": 1}
        ]
    })
    assert list(client.recordings()) == [{"id": "r1", "game_code": "aa00"}]


def test_recordings_accepts_nested_and_bare_list_shapes():
    nested = StubClient({
        "/v1/recordings": [
            {"data": {"items": [{"id": "nested"}], "total": 1}}
        ]
    })
    assert [item["id"] for item in nested.recordings()] == ["nested"]

    bare = StubClient({"/v1/recordings": [[{"id": "bare"}]]})
    assert [item["id"] for item in bare.recordings()] == ["bare"]


def test_empty_page_falls_back_to_manifest_and_filters_game():
    client = StubClient({
        "/v1/recordings": [{"items": [], "total": 2}],
        "/v1/dataset/manifest": [
            {
                "recordings": [
                    {"id": "a", "game_code": "aa00"},
                    {"id": "b", "game_code": "bb01"},
                ]
            }
        ],
    })
    assert list(client.recordings("aa00")) == [{"id": "a", "game_code": "aa00"}]


def test_unknown_success_payload_is_not_silently_treated_as_empty():
    client = StubClient({"/v1/recordings": [{"results": []}]})
    with pytest.raises(DatasetProtocolError, match="response keys: results"):
        list(client.recordings())


def test_recording_id_is_url_encoded():
    client = ArcDatasetClient("https://example.invalid/agi-3", api_key="secret")
    request = client._request("/v1/recordings/id%2Fwith%20space/download")
    assert request.full_url.endswith("/agi-3/v1/recordings/id%2Fwith%20space/download")
    assert request.headers["X-api-key"] == "secret"
