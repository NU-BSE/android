from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import BigInteger, DateTime, Index, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Recording(Base):
    __tablename__ = "recordings"
    __table_args__ = (
        UniqueConstraint("source_relpath", "sha256", name="uq_recording_source_sha"),
        Index("ix_recordings_game_code_started", "game_code", "started_at"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    game_code: Mapped[str] = mapped_column(String(32), index=True)
    file_name: Mapped[str] = mapped_column(String(512))
    source_relpath: Mapped[str] = mapped_column(String(1024))
    object_key: Mapped[str] = mapped_column(String(1024), unique=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    byte_size: Mapped[int] = mapped_column(BigInteger)
    record_count: Mapped[int] = mapped_column(Integer)
    transition_count: Mapped[int] = mapped_column(Integer)
    game_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    frame_shapes: Mapped[dict] = mapped_column(JSON, default=dict)
    available_actions: Mapped[list] = mapped_column(JSON, default=list)
    final_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
