from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecordingMetadata:
    sha256: str
    byte_size: int
    record_count: int
    transition_count: int
    game_id: str | None
    started_at: datetime | None
    ended_at: datetime | None
    frame_shapes: dict[str, int]
    available_actions: list[int]
    final_summary: dict[str, Any] | None


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def infer_shape(value: object) -> tuple[int, ...]:
    """Infer a rectangular nested-list shape; return () when it is ragged/non-list."""
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    child_shapes = [infer_shape(item) for item in value]
    first = child_shapes[0]
    if any(shape != first for shape in child_shapes[1:]):
        return ()
    return (len(value), *first)


def is_transition_record(data: object) -> bool:
    return isinstance(data, dict) and "game_id" in data and "frame" in data


def inspect_jsonl(path: str | Path) -> RecordingMetadata:
    source = Path(path)
    digest = hashlib.sha256()
    frame_shapes: Counter[str] = Counter()
    action_ids: set[int] = set()
    record_count = 0
    transition_count = 0
    game_id: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    final_summary: dict[str, Any] | None = None

    with source.open("rb") as raw:
        for line_number, raw_line in enumerate(raw, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            record_count += 1
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {source}:{line_number}: {exc.msg}"
                ) from exc

            timestamp = _parse_timestamp(record.get("timestamp"))
            if timestamp is not None:
                started_at = timestamp if started_at is None else min(started_at, timestamp)
                ended_at = timestamp if ended_at is None else max(ended_at, timestamp)

            data = record.get("data")
            if is_transition_record(data):
                transition_count += 1
                if game_id is None and isinstance(data.get("game_id"), str):
                    game_id = data["game_id"]

                shape = infer_shape(data.get("frame"))
                shape_key = "x".join(str(dimension) for dimension in shape) or "ragged"
                frame_shapes[shape_key] += 1

                available = data.get("available_actions")
                if isinstance(available, list):
                    action_ids.update(item for item in available if isinstance(item, int))
            elif isinstance(data, dict):
                # ARC recordings place aggregate play statistics in the final line.
                final_summary = data

    return RecordingMetadata(
        sha256=digest.hexdigest(),
        byte_size=source.stat().st_size,
        record_count=record_count,
        transition_count=transition_count,
        game_id=game_id,
        started_at=started_at,
        ended_at=ended_at,
        frame_shapes=dict(sorted(frame_shapes.items())),
        available_actions=sorted(action_ids),
        final_summary=final_summary,
    )
