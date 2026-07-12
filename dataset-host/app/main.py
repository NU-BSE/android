from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from botocore.exceptions import ClientError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth import require_read_access
from app.config import get_settings
from app.db import Base, engine, get_db
from app.models import Recording
from app.schemas import (
    DatasetStats,
    GameOut,
    ManifestOut,
    RecordingOut,
    RecordingPage,
    SignedUrlOut,
)
from app.storage import ObjectStorage

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    if settings.s3_create_bucket:
        ObjectStorage().ensure_bucket()
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["Content-Type", "Range", "X-API-Key"],
    expose_headers=["Accept-Ranges", "Content-Length", "Content-Range", "ETag"],
)


def _recording_out(recording: Recording) -> RecordingOut:
    item = RecordingOut.model_validate(recording)
    item.download_path = f"{settings.api_prefix}/recordings/{recording.id}/download"
    item.signed_url_path = f"{settings.api_prefix}/recordings/{recording.id}/signed-url"
    return item


def _get_recording(db: Session, recording_id: str) -> Recording:
    recording = db.get(Recording, recording_id)
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    return recording


def _stats(db: Session) -> DatasetStats:
    row = db.execute(
        select(
            func.count(func.distinct(Recording.game_code)),
            func.count(Recording.id),
            func.coalesce(func.sum(Recording.byte_size), 0),
            func.coalesce(func.sum(Recording.transition_count), 0),
        )
    ).one()
    return DatasetStats(
        games=int(row[0]),
        recordings=int(row[1]),
        total_bytes=int(row[2]),
        total_transitions=int(row[3]),
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    f"{settings.api_prefix}/stats",
    response_model=DatasetStats,
    dependencies=[Depends(require_read_access)],
)
def stats(db: Session = Depends(get_db)) -> DatasetStats:
    return _stats(db)


@app.get(
    f"{settings.api_prefix}/games",
    response_model=list[GameOut],
    dependencies=[Depends(require_read_access)],
)
def list_games(db: Session = Depends(get_db)) -> list[GameOut]:
    rows = db.execute(
        select(
            Recording.game_code,
            func.count(Recording.id),
            func.coalesce(func.sum(Recording.byte_size), 0),
            func.coalesce(func.sum(Recording.transition_count), 0),
        )
        .group_by(Recording.game_code)
        .order_by(Recording.game_code)
    ).all()
    return [
        GameOut(
            game_code=row[0],
            recording_count=int(row[1]),
            total_bytes=int(row[2]),
            total_transitions=int(row[3]),
        )
        for row in rows
    ]


@app.get(
    f"{settings.api_prefix}/recordings",
    response_model=RecordingPage,
    dependencies=[Depends(require_read_access)],
)
def list_recordings(
    game_code: str | None = None,
    limit: int = Query(default=100, ge=1),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> RecordingPage:
    limit = min(limit, settings.max_page_size)
    filters = [Recording.game_code == game_code] if game_code else []
    total = db.scalar(select(func.count(Recording.id)).where(*filters)) or 0
    recordings = db.scalars(
        select(Recording)
        .where(*filters)
        .order_by(Recording.game_code, Recording.started_at, Recording.file_name)
        .limit(limit)
        .offset(offset)
    ).all()
    return RecordingPage(
        items=[_recording_out(item) for item in recordings],
        total=int(total),
        limit=limit,
        offset=offset,
    )


@app.get(
    f"{settings.api_prefix}/recordings/{{recording_id}}",
    response_model=RecordingOut,
    dependencies=[Depends(require_read_access)],
)
def get_recording(recording_id: str, db: Session = Depends(get_db)) -> RecordingOut:
    return _recording_out(_get_recording(db, recording_id))


@app.get(
    f"{settings.api_prefix}/recordings/{{recording_id}}/signed-url",
    response_model=SignedUrlOut,
    dependencies=[Depends(require_read_access)],
)
def signed_url(recording_id: str, db: Session = Depends(get_db)) -> SignedUrlOut:
    recording = _get_recording(db, recording_id)
    url = ObjectStorage().presigned_get_url(
        recording.object_key, expires_in=settings.signed_url_ttl_seconds
    )
    return SignedUrlOut(url=url, expires_in=settings.signed_url_ttl_seconds)


@app.api_route(
    f"{settings.api_prefix}/recordings/{{recording_id}}/download",
    methods=["GET", "HEAD"],
    dependencies=[Depends(require_read_access)],
)
def download_recording(
    recording_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    recording = _get_recording(db, recording_id)
    byte_range = request.headers.get("range")
    if byte_range and not byte_range.startswith("bytes="):
        raise HTTPException(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            detail="Only byte ranges are supported",
        )

    storage = ObjectStorage()
    try:
        response = storage.get_object(recording.object_key, byte_range=byte_range)
    except ClientError as exc:
        error_code = str(exc.response.get("Error", {}).get("Code", ""))
        if error_code in {"InvalidRange", "416", "RequestedRangeNotSatisfiable"}:
            raise HTTPException(
                status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                detail="Requested byte range is outside the object",
            ) from exc
        if error_code in {"NoSuchKey", "404", "NotFound"}:
            raise HTTPException(status_code=404, detail="Recording object not found") from exc
        raise

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f'attachment; filename="{Path(recording.file_name).name}"',
        "ETag": response.get("ETag", ""),
    }
    if response.get("ContentLength") is not None:
        headers["Content-Length"] = str(response["ContentLength"])
    if response.get("ContentRange"):
        headers["Content-Range"] = response["ContentRange"]

    status_code = 206 if response.get("ContentRange") else 200
    if request.method == "HEAD":
        response["Body"].close()
        return Response(
            status_code=status_code,
            headers=headers,
            media_type="application/x-ndjson",
        )

    return StreamingResponse(
        storage.iter_body(response["Body"], settings.stream_chunk_bytes),
        status_code=status_code,
        media_type="application/x-ndjson",
        headers=headers,
    )


@app.get(
    f"{settings.api_prefix}/dataset/manifest",
    response_model=ManifestOut,
    dependencies=[Depends(require_read_access)],
)
def dataset_manifest(db: Session = Depends(get_db)) -> ManifestOut:
    recordings = db.scalars(
        select(Recording).order_by(Recording.game_code, Recording.started_at, Recording.file_name)
    ).all()
    return ManifestOut(
        generated_at=datetime.now(timezone.utc),
        stats=_stats(db),
        recordings=[_recording_out(item) for item in recordings],
    )
