#!/usr/bin/env python3
"""Fail closed when a normalized OS-settings artifact is unsafe for training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scrape_os_settings import audit_normalized_documents  # noqa: E402


def load_documents(path: Path) -> list[Mapping[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        documents = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("documents"), list):
        documents = payload["documents"]
    else:
        raise ValueError("artifact must be a JSON array or contain a documents array")
    if not all(isinstance(item, Mapping) for item in documents):
        raise ValueError("every normalized document must be an object")
    return documents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "documents",
        nargs="?",
        type=Path,
        default=Path("artifacts/normalized-documents.json"),
    )
    args = parser.parse_args(argv)
    try:
        report = audit_normalized_documents(load_documents(args.documents))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report.to_mapping(), ensure_ascii=False, indent=2, sort_keys=True))
    if (
        report.invalid_claim_count
        or report.duplicate_claim_count
        or report.duplicate_document_count
        or report.structural_error_count
    ):
        return 1
    if report.emitted_claim_count < 1:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
