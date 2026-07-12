import json
from pathlib import Path

from app.parser import infer_shape, inspect_jsonl


def test_infer_shape() -> None:
    assert infer_shape([[[1, 2], [3, 4]]]) == (1, 2, 2)
    assert infer_shape([[1], [1, 2]]) == ()


def test_inspect_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "recording.jsonl"
    rows = [
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "data": {
                "game_id": "aa00-1234",
                "frame": [[[1, 2], [3, 4]]],
                "available_actions": [1, 2, 3],
            },
        },
        {
            "timestamp": "2026-01-01T00:00:01+00:00",
            "data": {
                "game_id": "aa00-1234",
                "frame": [[[4, 3], [2, 1]]],
                "available_actions": [2, 4],
            },
        },
        {
            "timestamp": "2026-01-01T00:00:02+00:00",
            "data": {"played": 1, "won": 0, "total_actions": 2},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    metadata = inspect_jsonl(path)
    assert metadata.record_count == 3
    assert metadata.transition_count == 2
    assert metadata.game_id == "aa00-1234"
    assert metadata.frame_shapes == {"1x2x2": 2}
    assert metadata.available_actions == [1, 2, 3, 4]
    assert metadata.final_summary == {"played": 1, "won": 0, "total_actions": 2}
