"""Download Duck recordings and reconstruct dense local trajectories."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

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
    parser.add_argument("--base-url", default="http://hse.onl/agi-3")
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


def main() -> int:
    args = _parse_args()
    if args.max_seed <= 0:
        raise SystemExit("--max-seed must be positive")
    candidate_seeds = tuple(args.seed or range(args.max_seed))
    arcengine = load_arcengine()
    client = ArcDatasetClient(args.base_url, api_key=args.api_key)
    all_examples = []
    reports = []
    errors = []

    for metadata in _recordings(client, args):
        recording_id = str(metadata["id"])
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
                except ReplayAlignmentError:
                    continue
                candidates.append((len(report.mismatches), str(path), examples, report))
            if not candidates:
                raise ReplayAlignmentError(
                    f"{recording_id}: no environment/seed reproduced the trace"
                )
            _, _, examples, report = min(candidates, key=lambda item: (item[0], item[1]))
            all_examples.extend(examples)
            reports.append(report.as_dict())
            print(json.dumps(report.as_dict(), sort_keys=True))
        except Exception as exc:
            row = {
                "recording_id": recording_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(row)
            print(json.dumps(row, sort_keys=True))

    if not all_examples:
        raise SystemExit("No dense transitions were reconstructed")
    save_examples_npz(all_examples, args.output)
    payload = {
        "recordings": reports,
        "errors": errors,
        "dense_transitions": len(all_examples),
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary": payload}, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
