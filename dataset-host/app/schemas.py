from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RecordingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    game_code: str
    file_name: str
    source_relpath: str
    sha256: str
    byte_size: int
    record_count: int
    transition_count: int
    game_id: str | None
    started_at: datetime | None
    ended_at: datetime | None
    frame_shapes: dict[str, int]
    available_actions: list[int]
    final_summary: dict | None
    ingested_at: datetime
    download_path: str | None = None
    signed_url_path: str | None = None


class RecordingPage(BaseModel):
    items: list[RecordingOut]
    total: int
    limit: int
    offset: int


class GameOut(BaseModel):
    game_code: str
    recording_count: int
    total_bytes: int
    total_transitions: int


class DatasetStats(BaseModel):
    games: int
    recordings: int
    total_bytes: int
    total_transitions: int


class SignedUrlOut(BaseModel):
    url: str
    expires_in: int


class ManifestOut(BaseModel):
    generated_at: datetime
    stats: DatasetStats
    recordings: list[RecordingOut]
