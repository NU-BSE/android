"""Download hosted recordings and reconstruct dense local trajectories."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from drlqn.data.arc_dataset import ArcDatasetClient
from drlqn.env.official_loader import environment_files, load_arcengine, make_arcade_env
from drlqn.train.duck_dataset import save_examples_npz
from drlqn.train.duck_replay import (
    ReplayAlignmentError,
    parse_recording_events,
    replay_recording,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://hse.onl/agi-3")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ARC_DATASET_API_KEY"),
        help="Dataset API key; prefer ARC_DATASET_API_KEY instead of shell history",
    )
    parser.add_argument("--game", action="append", default=[])
    parser.add_argument("--recording", action="append", default=[])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("arc-prize-2026-arc-agi-3/environment_files"),
    )
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--max-seed", type=int, default=64)
    parser.add_argument("--allow-mismatch", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/duck_dense_transitions.npz"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/duck_replay_report.json"),
    )
    return parser.parse_args()


def _recordings(client: ArcDatasetClient, args: argparse.Namespace):
    if args.recording:
        for recording_id in args.recording:
            yield client.get_recording(recording_id)
        return
    games = args.game or [None]
    for game in games:
        yield from client.recordings(game_code=game)


def _error(kind: str, exc: Exception | str, **fields: Any) -> dict[str, Any]:
    message = str(exc)
    return {
        "kind": kind,
        **fields,
        "error": message,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _empty_reason(stats: dict[str, Any], metadata_count: int) -> str:
    advertised = stats.get("recordings")
    if isinstance(advertised, (int, float)) and int(advertised) == 0:
        return (
            "The hosted dataset API is reachable but its metadata index contains zero "
            "recordings. Ingest the recording dataset into the deployed PostgreSQL/S3 "
            "instance before running replay."
        )
    if isinstance(advertised, (int, float)) and int(advertised) > metadata_count:
        return (
            f"The API advertises {int(advertised)} recordings but the list and manifest "
            "endpoints returned none. This indicates a deployed API/proxy version mismatch."
        )
    if stats.get("error"):
        return (
            "Recording discovery returned no metadata and the stats preflight also failed; "
            "inspect the report for the HTTP/protocol error."
        )
    return "The recording list and manifest endpoints returned no metadata."


def main() -> int:
    args = _parse_args()
    if args.max_seed <= 0:
        raise SystemExit("--max-seed must be positive")
    candidate_seeds = tuple(args.seed or range(args.max_seed))
    client = ArcDatasetClient(args.base_url, api_key=args.api_key)

    try:
        dataset_stats = client.stats()
    except Exception as exc:
        dataset_stats = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps({"dataset_preflight": dataset_stats}, sort_keys=True), flush=True)

    discovery_errors: list[dict[str, Any]] = []
    try:
        metadata_rows = list(_recordings(client, args))
    except Exception as exc:
        metadata_rows = []
        discovery_errors.append(
            _error("recording_discovery", f"{type(exc).__name__}: {exc}")
        )

    base_report: dict[str, Any] = {
        "base_url": args.base_url,
        "dataset_stats": dataset_stats,
        "recordings_discovered": len(metadata_rows),
        "recordings": [],
        "errors": discovery_errors,
        "dense_transitions": 0,
        "output": str(args.output),
    }
    if not metadata_rows:
        reason = _empty_reason(dataset_stats, 0)
        base_report["failure"] = reason
        _write_report(args.report, base_report)
        print(json.dumps({"replay_failure": reason, "report": str(args.report)}), flush=True)
        raise SystemExit(reason)

    arcengine = load_arcengine()
    all_examples = []
    reports = []
    errors = list(discovery_errors)

    for metadata in metadata_rows:
        recording_id = str(metadata.get("id") or metadata.get("recording_id") or "")
        if not recording_id:
            row = _error(
                "metadata",
                "recording metadata has no id or recording_id field",
                metadata_keys=sorted(map(str, metadata.keys())),
            )
            errors.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            continue
        try:
            recording = parse_recording_events(
                client.events(recording_id),
                metadata=metadata,
            )
            paths = environment_files(args.root, recording.game_code)
            if not paths:
                raise FileNotFoundError(
                    f"no local environment for game {recording.game_code} under {args.root}"
                )

            candidates = []
            alignment_errors = []
            for path in paths:
                try:
                    examples, report = replay_recording(
                        recording,
                        env_factory=lambda seed, path=path: make_arcade_env(
                            path, arcengine, seed=seed
                        ),
                        candidate_seeds=candidate_seeds,
                        environment_name=str(path),
                        strict=not args.allow_mismatch,
                    )
                except ReplayAlignmentError as exc:
                    alignment_errors.append(f"{path}: {exc}")
                    continue
                candidates.append((len(report.mismatches), str(path), examples, report))
            if not candidates:
                detail = "; ".join(alignment_errors[:3])
                suffix = f"; first alignment errors: {detail}" if detail else ""
                raise ReplayAlignmentError(
                    f"{recording_id}: no environment/seed reproduced the trace{suffix}"
                )
            _, _, examples, report = min(candidates, key=lambda item: (item[0], item[1]))
            all_examples.extend(examples)
            reports.append(report.as_dict())
            print(json.dumps(report.as_dict(), sort_keys=True), flush=True)
        except Exception as exc:
            row = _error(
                "recording_replay",
                f"{type(exc).__name__}: {exc}",
                recording_id=recording_id,
                game_code=metadata.get("game_code"),
                transition_count=metadata.get("transition_count"),
                source_relpath=metadata.get("source_relpath"),
            )
            errors.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    payload = {
        **base_report,
        "recordings": reports,
        "errors": errors,
        "dense_transitions": len(all_examples),
        "recordings_replayed": len(reports),
        "recordings_failed": len(errors),
    }
    if not all_examples:
        first = errors[0]["error"] if errors else "no recording produced a transition"
        reason = (
            f"Discovered {len(metadata_rows)} recording(s), but none replayed successfully. "
            f"First failure: {first}"
        )
        payload["failure"] = reason
        _write_report(args.report, payload)
        print(json.dumps({"replay_failure": reason, "report": str(args.report)}), flush=True)
        raise SystemExit(reason)

    save_examples_npz(all_examples, args.output)
    _write_report(args.report, payload)
    print(json.dumps({"summary": payload}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
