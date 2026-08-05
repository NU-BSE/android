#!/usr/bin/env python3
"""Generate natural screen-search chat SFT records with Qwen3.7-Plus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from android_drlqn.qwen_screen_qa import (  # noqa: E402
    DEFAULT_API_KEY_ENV,
    DEFAULT_MODEL,
    QwenScreenQAClient,
    append_chat_jsonl,
    chat_sft_records,
    load_screen_contexts,
)


def _completed_screen_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict) and value.get("screen_id"):
            completed.add(str(value["screen_id"]))
    return completed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/training/chat-screen-qa-qwen.jsonl"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--questions-per-screen", type=int, default=4)
    parser.add_argument("--language", default="English")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-screens", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    client = QwenScreenQAClient(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        timeout_s=args.timeout,
        max_retries=args.max_retries,
    )
    completed = _completed_screen_ids(args.output) if args.resume else set()
    screens = generated = 0
    for context in load_screen_contexts(args.input):
        if context.screen_id in completed:
            continue
        if args.max_screens is not None and screens >= args.max_screens:
            break
        items = client.generate(
            context,
            count=args.questions_per_screen,
            language=args.language,
        )
        generated += append_chat_jsonl(
            args.output,
            chat_sft_records(context, items, model=args.model),
        )
        screens += 1

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "model": args.model,
        "screens": screens,
        "records": generated,
        "resume": args.resume,
        "api_key_written": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
