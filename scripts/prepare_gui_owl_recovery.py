#!/usr/bin/env python3
"""Prepare a filtered broad instruction corpus for structural student recovery.

The default source is the MIT-licensed HuggingFaceH4/ultrachat_200k train_sft
split at a pinned dataset revision. It is text-only by design; the recovery
trainer mixes these broad conversations with the existing multimodal GUI corpus
so structural recovery does not depend only on Android Settings examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_DATASET = "HuggingFaceH4/ultrachat_200k"
DEFAULT_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
DEFAULT_SPLIT = "train_sft"


def _clean_messages(value: object, *, max_turn_chars: int, max_total_chars: int) -> list[dict[str, str]] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    messages: list[dict[str, str]] = []
    total = 0
    expected = "user"
    for raw in value:
        if not isinstance(raw, Mapping):
            return None
        role = str(raw.get("role") or "")
        content = str(raw.get("content") or "").strip()
        if role != expected or not content or len(content) > max_turn_chars:
            return None
        total += len(content)
        if total > max_total_chars:
            return None
        messages.append({"role": role, "content": content})
        expected = "assistant" if expected == "user" else "user"
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return None
    return messages


def _record_id(messages: Sequence[Mapping[str, str]]) -> str:
    payload = json.dumps(list(messages), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare(args: argparse.Namespace) -> dict[str, object]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "the recovery preparer needs the data extra with datasets: "
            "python -m pip install -e '.[data]'"
        ) from exc

    dataset = load_dataset(
        args.dataset,
        split=args.split,
        revision=args.revision,
        streaming=True,
    )
    if args.shuffle_buffer > 1:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    accepted = rejected = 0
    seen: set[str] = set()
    with output.open("w", encoding="utf-8") as handle:
        for row in dataset:
            messages = _clean_messages(
                row.get("messages") if isinstance(row, Mapping) else None,
                max_turn_chars=args.max_turn_chars,
                max_total_chars=args.max_total_chars,
            )
            if messages is None:
                rejected += 1
                continue
            identifier = _record_id(messages)
            if identifier in seen:
                rejected += 1
                continue
            seen.add(identifier)
            record = {
                "messages": messages,
                "teacher_mode": "base",
                "recovery_source": {
                    "dataset": args.dataset,
                    "revision": args.revision,
                    "split": args.split,
                    "record_sha256": identifier,
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            accepted += 1
            if accepted >= args.max_records:
                break
    if accepted < args.min_records:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"only {accepted} valid recovery records were produced; "
            f"minimum required is {args.min_records}"
        )

    manifest = {
        "schema_version": 1,
        "dataset": args.dataset,
        "revision": args.revision,
        "split": args.split,
        "license_declared_by_source": "MIT" if args.dataset == DEFAULT_DATASET else "verify-source-license",
        "output": str(output),
        "accepted_records": accepted,
        "rejected_records_seen": rejected,
        "teacher_mode": "base",
        "text_only": True,
        "mixed_with_multimodal_specialized_corpora_during_recovery": True,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "max_turn_chars": args.max_turn_chars,
        "max_total_chars": args.max_total_chars,
        "credentials_written": False,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output", default="artifacts/training/gui-owl-recovery.jsonl")
    parser.add_argument("--max-records", type=int, default=50000)
    parser.add_argument("--min-records", type=int, default=10000)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--max-turn-chars", type=int, default=8000)
    parser.add_argument("--max-total-chars", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if min(args.max_records, args.min_records, args.max_turn_chars, args.max_total_chars) < 1:
        parser.error("record and character limits must be positive")
    if args.min_records > args.max_records:
        parser.error("min-records cannot exceed max-records")
    print(json.dumps(prepare(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
