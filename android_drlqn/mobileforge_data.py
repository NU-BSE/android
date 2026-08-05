"""Prepare safe GUI-Owl SFT records from MobileForge exploration JSON.

The release also contains compressed pickle objects. They are never opened here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

DATASET_ID = "lgy0404/mobileforge-exploration-trajectories"
DEFAULT_PACKAGE = "com.android.settings"
COORDINATE_SCALE = 1000
_FORBIDDEN_FILES = (".pkl", ".pickle", ".pkl.zst", ".pickle.zst")


@dataclass(frozen=True)
class ScreenRecord:
    record_id: str
    package: str
    goal: str
    screen_name: str
    screenshot: str
    visible_labels: tuple[str, ...]
    action_name: str
    action_arguments: Mapping[str, object]
    source_file: str


def download_mobileforge_settings(
    local_dir: str | Path,
    *,
    revision: str | None = None,
    include_raw_archives: bool = False,
) -> Path:
    """Download release metadata plus the parsed Android Settings subtree."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install the data extra: python -m pip install -e '.[data]'"
        ) from exc
    patterns = ["README.md", "manifest.json", f"final/{DEFAULT_PACKAGE}/**"]
    if include_raw_archives:
        patterns.append(f"raw/{DEFAULT_PACKAGE}/**")
    return Path(
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=revision,
            allow_patterns=patterns,
            local_dir=str(local_dir),
        )
    )


def _text(value: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return " ".join(item.split())
    return ""


def _labels(value: object, limit: int = 64) -> tuple[str, ...]:
    output: list[str] = []

    def visit(item: object) -> None:
        if len(output) >= limit:
            return
        if isinstance(item, str):
            text = " ".join(item.split())
            if text and text not in output:
                output.append(text)
        elif isinstance(item, Mapping):
            visit(
                item.get("text")
                or item.get("label")
                or item.get("content_desc")
                or item.get("name")
                or ""
            )
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(output)


def _point(value: object) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if 0 <= x <= 1 and 0 <= y <= 1:
        x, y = x * COORDINATE_SCALE, y * COORDINATE_SCALE
    return [
        max(0, min(COORDINATE_SCALE, round(x))),
        max(0, min(COORDINATE_SCALE, round(y))),
    ]


def _action(raw: object) -> tuple[str, dict[str, object]] | None:
    if isinstance(raw, str):
        kind, args = raw, {}
    elif isinstance(raw, Mapping):
        kind = str(
            raw.get("name")
            or raw.get("action_type")
            or raw.get("type")
            or raw.get("action")
            or ""
        )
        nested = raw.get("arguments")
        args = nested if isinstance(nested, Mapping) else raw
    else:
        return None
    lower = kind.casefold()
    if any(
        token in lower
        for token in ("adb", "shell", "root", "accessibility", "settings put")
    ):
        return None
    target = _text(args, "target_label", "label", "text", "description")
    if any(token in lower for token in ("tap", "click", "press")):
        coordinate = _point(
            args.get("coordinate")
            or args.get("point")
            or args.get("position")
            or args.get("xy")
        )
        if coordinate is None and args.get("x") is not None:
            coordinate = _point((args.get("x"), args.get("y")))
        if coordinate is None and not target:
            return None
        payload: dict[str, object] = {}
        if coordinate is not None:
            payload["coordinate"] = coordinate
        if target:
            payload["target_label"] = target
        return "mobile.click", payload
    if any(token in lower for token in ("swipe", "scroll")):
        return "mobile.swipe", {
            "start": _point(args.get("start")) or [500, 720],
            "end": _point(args.get("end")) or [500, 280],
            "duration": int(args.get("duration") or 300),
        }
    if "back" in lower:
        return "mobile.system_button", {"button": "Back"}
    if "wait" in lower or "sleep" in lower:
        return "mobile.wait", {"seconds": 1}
    return None


def _mappings(value: object) -> Iterator[Mapping[str, object]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _mappings(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _mappings(child)


def screen_records_from_json(
    value: object,
    *,
    source_file: str,
    dataset_root: str | Path,
    package: str = DEFAULT_PACKAGE,
) -> Iterator[ScreenRecord]:
    """Extract action-bearing screen rows from parsed release JSON."""
    root, seen = Path(dataset_root), set()
    for item in _mappings(value):
        item_package = _text(item, "package", "package_name", "app_package")
        if item_package and item_package != package:
            continue
        screenshot = _text(
            item,
            "screenshot",
            "screenshot_path",
            "image",
            "image_path",
            "before_screenshot",
            "screen_path",
        )
        if not screenshot:
            continue
        raw_action = next(
            (item[key] for key in ("action", "tool_call", "operation", "event") if key in item),
            None,
        )
        action = _action(raw_action)
        if action is None:
            continue
        action_name, arguments = action
        path = Path(screenshot)
        if not path.is_absolute():
            near = Path(source_file).parent / path
            path = near if near.exists() else root / path
        goal = _text(item, "goal", "instruction", "task", "exploration_goal", "query")
        action_name, arguments = action
        screen = _text(item, "screen_name", "page_name", "title", "activity")
        if not screen:
            screen = str(arguments.get("target_label") or "").strip()
        if not screen and goal:
            screen = goal[:160]
        labels = ()
        for key in ("visible_labels", "ui_texts", "texts", "elements", "ui_elements"):
            if key in item:
                labels = _labels(item[key])
                if labels:
                    break
        seed = json.dumps(
            [source_file, str(path), goal, screen, action_name, arguments],
            sort_keys=True,
            ensure_ascii=False,
        )
        record_id = hashlib.sha256(seed.encode()).hexdigest()[:24]
        if record_id in seen:
            continue
        seen.add(record_id)
        yield ScreenRecord(
            record_id,
            item_package or package,
            goal,
            screen,
            str(path),
            labels,
            action_name,
            arguments,
            source_file,
        )


def iter_mobileforge_screen_records(
    dataset_root: str | Path, *, package: str = DEFAULT_PACKAGE
) -> Iterator[ScreenRecord]:
    """Read final/<package> JSON/JSONL and skip all executable serialization."""
    root = Path(dataset_root)
    app_root = root / "final" / package
    if not app_root.is_dir():
        raise FileNotFoundError(f"parsed app directory not found: {app_root}")
    for path in sorted(app_root.rglob("*")):
        if any(path.name.casefold().endswith(suffix) for suffix in _FORBIDDEN_FILES):
            continue
        if path.suffix.casefold() == ".json":
            values = [json.loads(path.read_text(encoding="utf-8"))]
        elif path.suffix.casefold() == ".jsonl":
            values = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            continue
        for value in values:
            yield from screen_records_from_json(
                value, source_file=str(path), dataset_root=root, package=package
            )


def gui_owl_sft_record(record: ScreenRecord) -> dict[str, object]:
    goal = record.goal or f"Find and open the {record.screen_name} screen"
    call = json.dumps(
        {"name": record.action_name, "arguments": dict(record.action_arguments)},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "id": record.record_id,
        "dataset": DATASET_ID,
        "package": record.package,
        "source_file": record.source_file,
        "source_revision": "",
        "screen_name": record.screen_name,
        "goal": record.goal,
        "visible_labels": list(record.visible_labels),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are GUI-Owl-1.5 operating Android. Return one concise "
                    "Action conclusion and one bounded mobile tool call."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": record.screenshot},
                    {
                        "type": "text",
                        "text": (
                            f"Task: {goal}\nPackage: {record.package}\n"
                            f"Current screen: {record.screen_name}\n"
                            "Visible labels: "
                            + (", ".join(record.visible_labels) or "(not provided)")
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": (
                    f"Action: Use {record.action_name.rsplit('.', 1)[-1]} "
                    f"for the next verified step.\n<tool_call>{call}</tool_call>"
                ),
            },
        ],
    }


def write_gui_policy_jsonl(
    records: Iterable[ScreenRecord],
    output: str | Path,
    *,
    max_records: int | None = None,
) -> int:
    path, count = Path(output), 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if max_records is not None and count >= max_records:
                break
            handle.write(json.dumps(gui_owl_sft_record(record), ensure_ascii=False) + "\n")
            count += 1
    return count
