#!/usr/bin/env python3
"""Prepare GUI-Owl policy SFT JSONL from UI-MOPD Uni-GUI-OpenMobile."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from android_drlqn.unigui_openmobile import (  # noqa: E402
    DATASET_ID,
    DEFAULT_APP,
    download_unigui_openmobile,
    iter_unigui_screen_records,
    write_gui_policy_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("artifacts/uni-gui-openmobile"),
    )
    parser.add_argument("--revision", help="immutable dataset commit or tag")
    parser.add_argument("--app", default=DEFAULT_APP)
    parser.add_argument("--package")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--include-terminate", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/training/gui-policy-unigui-openmobile.jsonl"),
    )
    args = parser.parse_args(argv)

    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    if args.max_records is not None and args.max_records < 1:
        parser.error("--max-records must be positive")

    source = args.source_dir
    if source is None:
        source = download_unigui_openmobile(
            args.download_dir,
            revision=args.revision,
            app=args.app,
            package=args.package,
            max_episodes=args.max_episodes,
        )
    records = iter_unigui_screen_records(
        source,
        app=args.app,
        package=args.package,
        split=args.split,
        include_terminate=args.include_terminate,
    )
    count = write_gui_policy_jsonl(
        records,
        args.output,
        source_revision=args.revision or "",
        max_records=args.max_records,
    )
    if count < 1:
        parser.exit(
            2,
            "error: no reviewed Uni-GUI-OpenMobile action records were found. "
            "Confirm the app/package filter and that screenshots were downloaded.\n",
        )

    manifest = {
        "dataset_id": DATASET_ID,
        "revision": args.revision or "unpinned",
        "app": args.app,
        "package": args.package,
        "split": args.split,
        "source_dir": str(source),
        "output": str(args.output),
        "records": count,
        "reviewed_steps_only": True,
        "raw_code_fields_used": False,
        "coordinates": "normalized_0_1000",
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
