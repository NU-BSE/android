#!/usr/bin/env python3
"""Split a GUI-policy corpus into episode-disjoint train and holdout files.

The production evaluator splits by shuffling records, which leaks: 98.9% of the
records in the current 0.2 holdout come from an episode whose other steps are in
the training split, and a step's neighbours share its goal, its app and a nearly
identical screenshot. That is harmless while nothing is trained on the corpus and
worthless afterwards, so a split is made once, by episode, and pinned to a file.

`--pin-holdout` carries an existing holdout forward: those episodes stay held out
when the corpus is widened, so numbers stay comparable across corpus versions.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{number}: record must be an object")
        for field in ("id", "episode_id"):
            if not str(value.get(field) or ""):
                raise ValueError(f"{path}:{number}: record has no {field}")
        records.append(dict(value))
    if not records:
        raise ValueError(f"corpus is empty: {path}")
    return records


def _image_paths(record: Mapping[str, object]) -> list[str]:
    paths: list[str] = []
    for message in record.get("messages") or ():
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            continue
        for part in content:
            if isinstance(part, Mapping) and str(part.get("type")) == "image":
                image = str(part.get("image") or "")
                if image:
                    paths.append(image)
    return paths


def _pinned_episodes(path: Path | None) -> set[str]:
    if path is None:
        return set()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"pinned holdout must be an object: {path}")
    episodes = value.get("holdout_episode_ids")
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)):
        raise ValueError(f"pinned holdout has no holdout_episode_ids: {path}")
    return {str(item) for item in episodes}


def _write(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--holdout-out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pin-holdout",
        type=Path,
        help="manifest whose holdout_episode_ids must stay in the holdout",
    )
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="do not verify that every referenced screenshot resolves",
    )
    args = parser.parse_args(argv)

    if not 0.0 < args.holdout_fraction < 1.0:
        parser.error("--holdout-fraction must lie in (0, 1)")

    records = _load(args.corpus)
    by_episode: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    for record in records:
        by_episode[str(record["episode_id"])].append(record)

    pinned = _pinned_episodes(args.pin_holdout)
    known = set(by_episode)
    missing_pins = sorted(pinned - known)
    holdout_episodes = [episode for episode in sorted(known) if episode in pinned]
    held = sum(len(by_episode[episode]) for episode in holdout_episodes)

    remaining = [episode for episode in sorted(known) if episode not in pinned]
    random.Random(args.seed).shuffle(remaining)
    target = round(len(records) * args.holdout_fraction)
    for episode in remaining:
        if held >= target:
            break
        holdout_episodes.append(episode)
        held += len(by_episode[episode])

    holdout_set = set(holdout_episodes)
    holdout = [r for e in sorted(holdout_set) for r in by_episode[e]]
    train = [r for e in sorted(known - holdout_set) for r in by_episode[e]]

    if holdout_set & (known - holdout_set):
        raise RuntimeError("episode appears in both splits")
    overlap = {str(r["id"]) for r in holdout} & {str(r["id"]) for r in train}
    if overlap:
        raise RuntimeError(f"{len(overlap)} record ids appear in both splits")
    if not train or not holdout:
        raise RuntimeError("a split came out empty; lower --holdout-fraction")

    unresolved: list[str] = []
    if not args.skip_image_check:
        for record in records:
            for image in _image_paths(record):
                if not Path(image).is_file():
                    unresolved.append(image)
    if unresolved:
        sample = "\n- ".join(unresolved[:5])
        raise FileNotFoundError(
            f"{len(unresolved)} referenced screenshots do not resolve from "
            f"{Path.cwd()}:\n- {sample}\n"
            "Re-prepare with a --source-dir relative to the repository root, or "
            "rewrite the recorded prefix."
        )

    _write(args.train_out, train)
    _write(args.holdout_out, holdout)

    apps = collections.Counter(str(r.get("app") or "") for r in records)
    manifest = {
        "schema_version": 1,
        "corpus": str(args.corpus),
        "corpus_records": len(records),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "selection": {
            "unit": "episode_id",
            "seed": args.seed,
            "holdout_fraction": args.holdout_fraction,
            "pinned_from": str(args.pin_holdout) if args.pin_holdout else "",
            "pinned_episodes": len(pinned),
            "pinned_episodes_absent_from_corpus": missing_pins,
        },
        "train": {
            "path": str(args.train_out),
            "episodes": len(known) - len(holdout_set),
            "records": len(train),
        },
        "holdout": {
            "path": str(args.holdout_out),
            "episodes": len(holdout_set),
            "records": len(holdout),
            "episode_ids": sorted(holdout_set),
            "holdout_episode_ids": sorted(holdout_set),
            "ids": [str(r["id"]) for r in holdout],
        },
        "episode_disjoint": True,
        "images_verified": not args.skip_image_check,
        "apps": dict(apps.most_common()),
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    summary = dict(manifest)
    summary["holdout"] = {k: v for k, v in manifest["holdout"].items() if k != "ids"}
    summary["holdout"].pop("episode_ids", None)
    summary["holdout"].pop("holdout_episode_ids", None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if missing_pins:
        print(
            f"warning: {len(missing_pins)} pinned episodes are absent from this corpus",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
