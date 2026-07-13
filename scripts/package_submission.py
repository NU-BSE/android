"""Create a deterministic offline submission bundle after a measurable-gain gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("notebooks/simplified-submission-approach.ipynb"),
    )
    parser.add_argument("--minimum-held-out-gain", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/drlqn_submission.zip"),
    )
    return parser.parse_args()


def _gain(payload) -> float | None:
    for key in (
        "held_out_level_gain",
        "mean_levels_completed_gain",
        "level_completion_gain",
    ):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, (int, float)):
            return float(value)
    if isinstance(payload, dict):
        for value in payload.values():
            result = _gain(value) if isinstance(value, dict) else None
            if result is not None:
                return result
    return None


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    for path in (args.checkpoint, args.metrics, args.notebook):
        if not path.exists():
            raise SystemExit(f"missing required file: {path}")
    wheels = sorted(args.wheelhouse.glob("*.whl"))
    if not wheels:
        raise SystemExit(f"wheelhouse has no offline wheels: {args.wheelhouse}")
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    gain = _gain(metrics)
    if gain is None or gain < args.minimum_held_out_gain:
        raise SystemExit(
            f"submission gate failed: held-out level gain {gain!r} is below "
            f"{args.minimum_held_out_gain}"
        )

    root = Path(__file__).resolve().parents[1]
    files = [
        *sorted((root / "drlqn").rglob("*.py")),
        root / "scripts" / "run_submission_scorecard.py",
        args.notebook,
        args.metrics,
        args.checkpoint,
        args.checkpoint.with_suffix(args.checkpoint.suffix + ".json"),
        *wheels,
    ]
    files = [path.resolve() for path in files if path.exists()]
    root = root.resolve()
    wheels = [path.resolve() for path in wheels]
    manifest = {
        "checkpoint": args.checkpoint.name,
        "notebook": args.notebook.name,
        "metrics": args.metrics.name,
        "held_out_level_gain": gain,
        "minimum_held_out_gain": args.minimum_held_out_gain,
        "files": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in files:
            if path.is_relative_to(root):
                name = path.relative_to(root).as_posix()
            elif path in wheels:
                name = f"wheelhouse/{path.name}"
            elif path == args.metrics.resolve():
                name = f"metrics/{path.name}"
            else:
                name = f"checkpoint/{path.name}"
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            data = path.read_bytes()
            archive.writestr(info, data)
            manifest["files"][name] = hashlib.sha256(data).hexdigest()
        info = zipfile.ZipInfo(
            "manifest.json",
            date_time=(2026, 1, 1, 0, 0, 0),
        )
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(
            info,
            json.dumps(manifest, indent=2, sort_keys=True).encode(),
        )
    print(
        json.dumps(
            {
                "bundle": str(args.output),
                "sha256": _sha(args.output),
                "gain": gain,
                "notebook": str(args.notebook),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
