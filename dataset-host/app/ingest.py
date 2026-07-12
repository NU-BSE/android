from __future__ import annotations

import argparse
import re
from pathlib import Path

from sqlalchemy import select

from app.config import get_settings
from app.db import Base, SessionLocal, engine
from app.models import Recording
from app.parser import inspect_jsonl
from app.storage import ObjectStorage

GAME_CODE_PATTERN = re.compile(r"^[A-Za-z]{2}\d{2}$")


def discover_jsonl(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def infer_game_code(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    for part in reversed(relative.parts[:-1]):
        if GAME_CODE_PATTERN.fullmatch(part):
            return part.lower()
    # Fallback keeps nonstandard folders ingestible while making them visible in metadata.
    return path.parent.name.lower()


def ingest_one(path: Path, root: Path, storage: ObjectStorage, force: bool = False) -> str:
    metadata = inspect_jsonl(path)
    source_relpath = path.relative_to(root).as_posix()
    game_code = infer_game_code(path, root)
    safe_relpath = source_relpath.replace("/", "__")
    object_key = f"raw/{game_code}/{metadata.sha256[:16]}/{safe_relpath}"

    with SessionLocal() as db:
        existing = db.scalar(
            select(Recording).where(
                Recording.source_relpath == source_relpath,
                Recording.sha256 == metadata.sha256,
            )
        )
        if existing and not force:
            return f"SKIP {source_relpath} ({existing.id})"

        if not storage.exists(object_key) or force:
            storage.upload_file(path, object_key, metadata.sha256)

        if existing:
            recording = existing
        else:
            recording = Recording(
                game_code=game_code,
                file_name=path.name,
                source_relpath=source_relpath,
                object_key=object_key,
                sha256=metadata.sha256,
                byte_size=metadata.byte_size,
                record_count=metadata.record_count,
                transition_count=metadata.transition_count,
                game_id=metadata.game_id,
                started_at=metadata.started_at,
                ended_at=metadata.ended_at,
                frame_shapes=metadata.frame_shapes,
                available_actions=metadata.available_actions,
                final_summary=metadata.final_summary,
            )
            db.add(recording)

        db.commit()
        db.refresh(recording)
        return f"OK   {source_relpath} -> {object_key} ({recording.id})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload ARC JSONL recordings to S3-compatible storage and index them in PostgreSQL."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--force", action="store_true", help="Re-upload matching objects")
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root is not a directory: {root}")

    Base.metadata.create_all(bind=engine)
    settings = get_settings()
    storage = ObjectStorage(settings)
    if settings.s3_create_bucket:
        storage.ensure_bucket()

    files = discover_jsonl(root)
    if not files:
        raise SystemExit(f"No .jsonl files found under {root}")

    failures = 0
    for path in files:
        try:
            print(ingest_one(path, root, storage, force=args.force), flush=True)
        except Exception as exc:  # continue ingesting independent recordings
            failures += 1
            print(f"FAIL {path.relative_to(root)}: {exc}", flush=True)

    print(f"Processed {len(files)} files with {failures} failure(s).")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
