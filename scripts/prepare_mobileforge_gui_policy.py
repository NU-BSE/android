#!/usr/bin/env python3
"""Prepare GUI-Owl policy SFT JSONL from MobileForge Android Settings data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from android_drlqn.mobileforge_data import (  # noqa: E402
    DATASET_ID,
    DEFAULT_PACKAGE,
    download_mobileforge_settings,
    iter_mobileforge_screen_records,
    write_gui_policy_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("artifacts/mobileforge-exploration-trajectories"),
    )
    parser.add_argument("--revision", help="immutable dataset commit or tag")
    parser.add_argument("--package", default=DEFAULT_PACKAGE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/training/gui-policy-mobileforge.jsonl"),
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--include-raw-archives",
        action="store_true",
        help="download raw archives but never deserialize pickle content",
    )
    args = parser.parse_args(argv)

    source = args.source_dir
    if source is None:
        source = download_mobileforge_settings(
            args.download_dir,
            revision=args.revision,
            include_raw_archives=args.include_raw_archives,
        )
    records = iter_mobileforge_screen_records(source, package=args.package)
    count = write_gui_policy_jsonl(records, args.output, max_records=args.max_records)
    if count < 1:
        parser.exit(
            2,
            "error: no action-bearing parsed JSON/JSONL records were found. "
            "MobileForge summary files are metadata; use the parsed final artifacts.\n",
        )

    manifest = {
        "dataset_id": DATASET_ID,
        "revision": args.revision or "unpinned",
        "package": args.package,
        "source_dir": str(source),
        "output": str(args.output),
        "records": count,
        "raw_pickle_loaded": False,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
