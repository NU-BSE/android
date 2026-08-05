"""Prepare GUI-Owl SFT records from UI-MOPD/Uni-GUI-OpenMobile.

The dataset already stores normalized ``task.json`` trajectories and per-step
screenshots. This module ignores raw generation/code fields and consumes only the
reviewed ``mobile_use`` plans and local screenshot references.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

DATASET_ID = "UI-MOPD/Uni-GUI-OpenMobile"
DEFAULT_APP = "settings"
COORDINATE_SCALE = 1000
_ALLOWED_PLAN_NAME = "mobile_use"
_QUOTED = re.compile(r"['\"]([^'\"]{1,120})['\"]")
_FORBIDDEN_TEXT = (
    "adb shell",
    "settings put",
    "dispatchgesture",
    "accessibilityservice",
    "devicepolicymanager",
    "su -c",
)


@dataclass(frozen=True)
class ScreenRecord:
    record_id: str
    episode_id: str
    step_index: int
    app: str
    package: str
    goal: str
    screen_name: str
    screenshot: str
    visible_labels: tuple[str, ...]
    action_summary: str
    action_name: str
    action_arguments: Mapping[str, object]
    target_bbox: tuple[int, int, int, int] | None
    source_file: str
    split: str


def _clean_text(value: object, *, limit: int = 2000) -> str:
    return " ".join(str(value or "").split())[:limit].strip()


def _norm(value: object) -> str:
    return _clean_text(value, limit=300).casefold()


def _matches_episode(
    payload: Mapping[str, object], *, app: str | None, package: str | None
) -> bool:
    if app and _norm(payload.get("app")) != _norm(app):
        return False
    if package:
        packages = [payload.get("app_package")]
        raw = payload.get("app_packages")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            packages.extend(raw)
        if _norm(package) not in {_norm(value) for value in packages if value}:
            return False
    return True


def download_unigui_openmobile(
    local_dir: str | Path,
    *,
    revision: str | None = None,
    app: str | None = DEFAULT_APP,
    package: str | None = None,
    max_episodes: int | None = None,
) -> Path:
    """Download task metadata, then screenshots only for selected episodes.

    The first snapshot fetches all small ``task.json`` files. After filtering by
    app/package, a second snapshot fetches only the selected episode screenshots,
    avoiding an unconditional 9+ GB download when only Settings is needed.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install the data extra: python -m pip install -e '.[data]'"
        ) from exc

    destination = Path(local_dir)
    root = Path(
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=revision,
            allow_patterns=["README.md", "*/task.json"],
            local_dir=str(destination),
        )
    )
    selected: list[str] = []
    for task_path in sorted(root.glob("*/task.json")):
        payload = json.loads(task_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            continue
        if not _matches_episode(payload, app=app, package=package):
            continue
        selected.append(task_path.parent.name)
        if max_episodes is not None and len(selected) >= max_episodes:
            break
    if not selected:
        raise FileNotFoundError(
            f"no Uni-GUI-OpenMobile episodes matched app={app!r}, package={package!r}"
        )
    patterns: list[str] = []
    for episode in selected:
        patterns.extend((f"{episode}/task.json", f"{episode}/screenshot_step*.png"))
    return Path(
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=revision,
            allow_patterns=patterns,
            local_dir=str(destination),
        )
    )


def _point(value: object, resolution: Sequence[object] | None) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not (x == x and y == y):
        return None
    if 0 <= x <= 1 and 0 <= y <= 1:
        x *= COORDINATE_SCALE
        y *= COORDINATE_SCALE
    elif x > COORDINATE_SCALE or y > COORDINATE_SCALE:
        if not isinstance(resolution, Sequence) or len(resolution) < 2:
            return None
        try:
            width, height = float(resolution[0]), float(resolution[1])
        except (TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        x = x / width * COORDINATE_SCALE
        y = y / height * COORDINATE_SCALE
    return [
        max(0, min(COORDINATE_SCALE, round(x))),
        max(0, min(COORDINATE_SCALE, round(y))),
    ]


def _bbox(value: object, resolution: Sequence[object] | None) -> tuple[int, int, int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) == 2:
        p0, p1 = _point(value[0], resolution), _point(value[1], resolution)
    elif len(value) == 4:
        p0, p1 = _point(value[:2], resolution), _point(value[2:], resolution)
    else:
        return None
    if p0 is None or p1 is None:
        return None
    x0, y0 = p0
    x1, y1 = p1
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1)


def _assert_safe_text(*values: object) -> None:
    joined = " ".join(_clean_text(value, limit=4000) for value in values).casefold()
    if any(marker in joined for marker in _FORBIDDEN_TEXT):
        raise ValueError("trajectory contains executable or privileged guidance")


def _button(value: object) -> str | None:
    normalized = _norm(value)
    mapping = {
        "back": "Back",
        "navigate_back": "Back",
        "home": "Home",
        "navigate_home": "Home",
        "menu": "Menu",
        "enter": "Enter",
        "keyboard_enter": "Enter",
    }
    return mapping.get(normalized)


def _action(
    plan: object,
    *,
    resolution: Sequence[object] | None,
) -> tuple[str, dict[str, object]] | None:
    if not isinstance(plan, Mapping) or _norm(plan.get("name")) != _ALLOWED_PLAN_NAME:
        return None
    raw_args = plan.get("arguments")
    if not isinstance(raw_args, Mapping):
        return None
    kind = _norm(raw_args.get("action"))
    _assert_safe_text(kind, raw_args.get("text"), raw_args.get("code"))

    if kind in {"click", "tap"}:
        coordinate = _point(raw_args.get("coordinate"), resolution)
        return ("mobile.click", {"coordinate": coordinate}) if coordinate else None
    if kind == "long_press":
        coordinate = _point(raw_args.get("coordinate"), resolution)
        if coordinate is None:
            return None
        return "mobile.long_press", {
            "coordinate": coordinate,
            "time": float(raw_args.get("time") or 1.0),
        }
    if kind == "swipe":
        start = _point(raw_args.get("coordinate"), resolution)
        end = _point(raw_args.get("coordinate2"), resolution)
        if start is None or end is None:
            return None
        return "mobile.swipe", {"coordinate": start, "coordinate2": end}
    if kind in {"type", "input_text"}:
        text = _clean_text(raw_args.get("text"), limit=1000)
        return ("mobile.type", {"text": text}) if text else None
    if kind in {"system_button", "key", "navigate_back", "navigate_home", "keyboard_enter"}:
        button = _button(raw_args.get("button") or kind)
        return ("mobile.system_button", {"button": button}) if button else None
    if kind == "wait":
        return "mobile.wait", {"time": float(raw_args.get("time") or 1.0)}
    if kind in {"answer", "ask_user"}:
        text = _clean_text(raw_args.get("text"), limit=1200)
        return (f"mobile.{kind}", {"text": text}) if text else None
    if kind in {"terminate", "status"}:
        status = _norm(raw_args.get("status") or raw_args.get("goal_status") or "success")
        if status not in {"success", "failure"}:
            return None
        return "mobile.terminate", {"status": status}
    return None


def _labels(step: Mapping[str, object], *, app: str) -> tuple[str, ...]:
    output: list[str] = []
    for source in (step.get("action"), step.get("thought"), step.get("query")):
        text = _clean_text(source, limit=3000)
        for label in _QUOTED.findall(text):
            label = _clean_text(label, limit=120)
            if label and label.casefold() not in {item.casefold() for item in output}:
                output.append(label)
                if len(output) >= 24:
                    return tuple(output)
    if app and app.casefold() != "settings":
        output.append(app)
    return tuple(output)


def iter_unigui_screen_records(
    dataset_root: str | Path,
    *,
    app: str | None = DEFAULT_APP,
    package: str | None = None,
    split: str | None = "train",
    include_terminate: bool = False,
    require_reviewed: bool = True,
    require_images: bool = True,
) -> Iterator[ScreenRecord]:
    """Yield reviewed, executable step records from normalized episode folders."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {root}")
    seen: set[str] = set()
    for task_path in sorted(root.glob("*/task.json")):
        payload = json.loads(task_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            continue
        if _norm(payload.get("device")) not in {"", "mobile"}:
            continue
        if bool(payload.get("is_delete")) or bool(payload.get("is_mock")):
            continue
        if not _matches_episode(payload, app=app, package=package):
            continue
        episode_id = _clean_text(payload.get("episode_id") or task_path.parent.name, limit=200)
        episode_app = _clean_text(payload.get("app") or app or "mobile", limit=200)
        episode_package = _clean_text(payload.get("app_package") or package or "unknown", limit=300)
        goal = _clean_text(payload.get("query"), limit=2000)
        resolution = payload.get("screen_resolution")
        steps = payload.get("data")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            continue
        for ordinal, raw_step in enumerate(steps):
            if not isinstance(raw_step, Mapping):
                continue
            if raw_step.get("is_use") is False or bool(raw_step.get("is_delete")):
                continue
            if require_reviewed and raw_step.get("is_reviewed") is False:
                continue
            step_split = _clean_text(raw_step.get("train_test") or "train", limit=50)
            if split and _norm(step_split) != _norm(split):
                continue
            parsed = _action(raw_step.get("plan"), resolution=resolution if isinstance(resolution, Sequence) else None)
            if parsed is None:
                continue
            action_name, arguments = parsed
            if action_name == "mobile.terminate" and not include_terminate:
                continue
            screenshot_name = _clean_text(raw_step.get("screenshot"), limit=500)
            if not screenshot_name:
                continue
            screenshot_path = task_path.parent / screenshot_name
            if require_images and not screenshot_path.is_file():
                raise FileNotFoundError(f"referenced screenshot not found: {screenshot_path}")
            step_index = int(raw_step.get("step") if raw_step.get("step") is not None else ordinal)
            action_summary = _clean_text(raw_step.get("action"), limit=1000)
            _assert_safe_text(action_summary)
            labels = _labels(raw_step, app=episode_app)
            screen_name = labels[0] if labels else f"{episode_app} step {step_index}"
            target_bbox = _bbox(raw_step.get("bbox"), resolution if isinstance(resolution, Sequence) else None)
            seed = json.dumps(
                [DATASET_ID, episode_id, step_index, screenshot_name, action_name, arguments],
                ensure_ascii=False,
                sort_keys=True,
            )
            record_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
            if record_id in seen:
                continue
            seen.add(record_id)
            yield ScreenRecord(
                record_id=record_id,
                episode_id=episode_id,
                step_index=step_index,
                app=episode_app,
                package=episode_package,
                goal=goal or _clean_text(raw_step.get("query"), limit=2000),
                screen_name=screen_name,
                screenshot=str(screenshot_path),
                visible_labels=labels,
                action_summary=action_summary,
                action_name=action_name,
                action_arguments=arguments,
                target_bbox=target_bbox,
                source_file=str(task_path),
                split=step_split,
            )


def gui_owl_sft_record(record: ScreenRecord, *, source_revision: str = "") -> dict[str, object]:
    call = json.dumps(
        {"name": record.action_name, "arguments": dict(record.action_arguments)},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    action_text = record.action_summary or f"Execute {record.action_name.rsplit('.', 1)[-1]}."
    return {
        "id": record.record_id,
        "dataset": DATASET_ID,
        "episode_id": record.episode_id,
        "step_index": record.step_index,
        "app": record.app,
        "package": record.package,
        "source_file": record.source_file,
        "source_revision": source_revision,
        "screen_name": record.screen_name,
        "goal": record.goal,
        "visible_labels": list(record.visible_labels),
        "target_bbox": list(record.target_bbox) if record.target_bbox else None,
        "split": record.split,
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
                            f"Task: {record.goal}\nApp: {record.app}\nPackage: {record.package}\n"
                            f"Current screen: {record.screen_name}\n"
                            "Visible labels: "
                            + (", ".join(record.visible_labels) or "(not provided)")
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": f"Action: {action_text}\n<tool_call>{call}</tool_call>",
            },
        ],
    }


def write_gui_policy_jsonl(
    records: Iterable[ScreenRecord],
    output: str | Path,
    *,
    source_revision: str = "",
    max_records: int | None = None,
) -> int:
    path, count = Path(output), 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if max_records is not None and count >= max_records:
                break
            handle.write(
                json.dumps(
                    gui_owl_sft_record(record, source_revision=source_revision),
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


__all__ = [
    "COORDINATE_SCALE",
    "DATASET_ID",
    "DEFAULT_APP",
    "ScreenRecord",
    "download_unigui_openmobile",
    "gui_owl_sft_record",
    "iter_unigui_screen_records",
    "write_gui_policy_jsonl",
]
