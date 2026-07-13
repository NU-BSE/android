"""Reconstruct dense Duck trajectories by replaying batched actions locally."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .duck_dataset import DuckExample, actions_from_python, normalize_action

Action = tuple[int, int | None, int | None]


@dataclass(frozen=True)
class Observation:
    grid: np.ndarray
    event_index: int
    level: int | None = None
    state: str = ""


@dataclass(frozen=True)
class ActionBatch:
    start: Observation
    actions: tuple[Action, ...]
    expected: Observation | None
    reasoning: str = ""


@dataclass(frozen=True)
class DuckRecording:
    recording_id: str
    game_code: str
    game_id: str
    batches: tuple[ActionBatch, ...]
    final_summary: dict[str, Any]

    @property
    def initial_grid(self) -> np.ndarray:
        if not self.batches:
            raise ValueError(f"{self.recording_id}: no replayable action batches")
        return self.batches[0].start.grid


@dataclass(frozen=True)
class AlignmentMismatch:
    batch_index: int
    event_index: int
    differing_cells: int
    expected_sha1: str
    actual_sha1: str


@dataclass(frozen=True)
class ReplayReport:
    recording_id: str
    game_code: str
    game_id: str
    environment: str
    seed: int
    actions: int
    batches: int
    verified_anchors: int
    mismatches: tuple[AlignmentMismatch, ...]
    final_state: str
    success: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "game_code": self.game_code,
            "game_id": self.game_id,
            "environment": self.environment,
            "seed": self.seed,
            "actions": self.actions,
            "batches": self.batches,
            "verified_anchors": self.verified_anchors,
            "mismatches": [mismatch.__dict__ for mismatch in self.mismatches],
            "final_state": self.final_state,
            "success": self.success,
        }


class ReplayAlignmentError(RuntimeError):
    pass


def _grid(value: Any) -> np.ndarray | None:
    if not isinstance(value, list):
        return None
    try:
        array = np.asarray(value, dtype=np.uint8)
    except (TypeError, ValueError):
        return None
    while array.ndim > 2:
        array = array[-1]
    if array.ndim != 2:
        return None
    if array.shape == (64, 64):
        return array.copy()
    if not (1 <= array.shape[0] <= 64 and 1 <= array.shape[1] <= 64):
        return None
    padded = np.zeros((64, 64), dtype=np.uint8)
    padded[: array.shape[0], : array.shape[1]] = array
    return padded


def _observation(payload: dict[str, Any], event_index: int) -> Observation | None:
    containers = [payload]
    data = payload.get("data")
    if isinstance(data, dict):
        containers.insert(0, data)
    for container in containers:
        for key in ("grid", "frame"):
            array = _grid(container.get(key))
            if array is None:
                continue
            level_raw = container.get("level", container.get("current_level"))
            try:
                level = int(level_raw) if level_raw is not None else None
            except (TypeError, ValueError):
                level = None
            state = str(container.get("state") or container.get("game_state") or "")
            return Observation(array, event_index, level=level, state=state)
    return None


def _python_code(tool_call: dict[str, Any]) -> str | None:
    function = tool_call.get("function") or {}
    if str(function.get("name") or tool_call.get("name") or "") != "python":
        return None
    arguments = function.get("arguments", tool_call.get("arguments"))
    if isinstance(arguments, dict):
        payload = arguments
    else:
        try:
            payload = json.loads(str(arguments or "{}"))
        except json.JSONDecodeError:
            return None
    code = payload.get("code")
    return str(code) if code is not None else None


def _actions(payload: dict[str, Any]) -> list[Action]:
    result: list[Action] = []
    for tool_call in payload.get("tool_calls") or []:
        if isinstance(tool_call, dict):
            code = _python_code(tool_call)
            if code:
                result.extend(actions_from_python(code))
    if result:
        # Tool-call Python is the authoritative Duck batch. Returning here avoids
        # double-counting mirrored action fields while preserving repeated moves.
        return result

    data = payload.get("data")
    containers = [payload]
    if isinstance(data, dict):
        containers.insert(0, data)
    for container in containers:
        raw_items: list[Any] = []
        if "actions" in container and isinstance(container["actions"], list):
            raw_items.extend(container["actions"])
        for key in ("action_input", "action", "action_id"):
            if key in container:
                raw_items.append(container[key])
        for raw in raw_items:
            normalized = normalize_action(raw)
            if normalized is not None:
                result.append(normalized)
        if result:
            # Prefer the nested data payload over a duplicate top-level mirror.
            break
    return result


def _reasoning(payload: dict[str, Any]) -> str:
    return str(payload.get("reasoning") or payload.get("content") or "")


def _stream_items(events: Iterable[dict[str, Any]]) -> Iterable[tuple[int, dict[str, Any]]]:
    for event_index, event in enumerate(events):
        messages = event.get("messages") if isinstance(event, dict) else None
        if isinstance(messages, list):
            for sub_index, message in enumerate(messages):
                if isinstance(message, dict):
                    # Stable compound index; keeps message order without requiring
                    # a second field in the training format.
                    yield event_index * 1_000_000 + sub_index, message
        elif isinstance(event, dict):
            yield event_index, event


def parse_recording_events(
    events: Iterable[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> DuckRecording:
    """Turn streamed dataset events into observed action batches.

    The parser accepts both inference-trace messages and ARC recording rows. For
    rows containing an action and its resulting frame, the action is treated as
    occurring before the frame. For chat-style messages, an observed grid occurs
    before the assistant's Python action call.
    """
    metadata = dict(metadata or {})
    recording_id = str(metadata.get("id") or metadata.get("recording_id") or "recording")
    game_id = str(metadata.get("game_id") or "")
    game_code = str(metadata.get("game_code") or "")
    current: Observation | None = None
    pending: list[Action] = []
    pending_reasoning: list[str] = []
    batches: list[ActionBatch] = []
    final_summary: dict[str, Any] = {}

    for event_index, payload in _stream_items(events):
        data = payload.get("data") if isinstance(payload.get("data"), dict) else None
        if not game_id:
            for container in (data, payload):
                if isinstance(container, dict) and isinstance(container.get("game_id"), str):
                    game_id = str(container["game_id"])
                    break
        observation = _observation(payload, event_index)
        actions = _actions(payload)
        role = str(payload.get("role") or "")
        transition_row = bool(data and "frame" in data and actions)

        def add_actions() -> None:
            if actions:
                pending.extend(actions)
                text = _reasoning(payload)
                if text:
                    pending_reasoning.append(text)

        def add_observation() -> None:
            nonlocal current, pending, pending_reasoning
            if observation is None:
                return
            if current is None:
                current = observation
                return
            if pending:
                batches.append(
                    ActionBatch(
                        start=current,
                        actions=tuple(pending),
                        expected=observation,
                        reasoning="\n".join(pending_reasoning),
                    )
                )
                pending = []
                pending_reasoning = []
            current = observation

        if transition_row:
            add_actions()
            add_observation()
        elif role == "assistant":
            add_observation()
            add_actions()
        else:
            add_observation()
            add_actions()

        if data is not None and observation is None and not actions:
            final_summary = dict(data)

    if current is not None and pending:
        batches.append(
            ActionBatch(
                start=current,
                actions=tuple(pending),
                expected=None,
                reasoning="\n".join(pending_reasoning),
            )
        )
    if not game_code and game_id:
        game_code = game_id.split("-", 1)[0].split("/", 1)[0]
    if not game_code:
        raise ValueError(f"{recording_id}: could not determine game code")
    if not game_id:
        game_id = game_code
    if not batches:
        raise ValueError(f"{recording_id}: no observed action batches found")
    return DuckRecording(
        recording_id=recording_id,
        game_code=game_code,
        game_id=game_id,
        batches=tuple(batches),
        final_summary=final_summary,
    )


def _digest(grid: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(grid, dtype=np.uint8).tobytes()).hexdigest()[:16]


def _same_grid(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and bool(np.array_equal(left, right))


def identify_seed(
    recording: DuckRecording,
    env_factory: Callable[[int], Any],
    candidate_seeds: Sequence[int],
) -> tuple[int, Any]:
    best: tuple[int, Any] | None = None
    for seed in candidate_seeds:
        env = env_factory(int(seed))
        frame = env.reset()
        if _same_grid(frame.grid, recording.initial_grid):
            candidate = (int(seed), env)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        raise ReplayAlignmentError(
            f"{recording.recording_id}: no local seed matches initial grid "
            f"{_digest(recording.initial_grid)}"
        )
    # Deterministic games often match every seed; the smallest seed is canonical.
    return best


def _summary_success(summary: dict[str, Any]) -> bool | None:
    for key in ("won", "win", "success"):
        value = summary.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
    return None


def replay_recording(
    recording: DuckRecording,
    env_factory: Callable[[int], Any],
    candidate_seeds: Sequence[int] = tuple(range(64)),
    environment_name: str = "local",
    strict: bool = True,
) -> tuple[list[DuckExample], ReplayReport]:
    """Replay every batched action and emit all intermediate transitions."""
    seed, _ = identify_seed(recording, env_factory, candidate_seeds)
    # Recreate the environment so seed identification cannot consume state.
    env = env_factory(seed)
    frame = env.reset()
    if not _same_grid(frame.grid, recording.initial_grid):
        raise ReplayAlignmentError("seed match was not reproducible")

    examples: list[DuckExample] = []
    mismatches: list[AlignmentMismatch] = []
    verified_anchors = 1
    action_index = 0
    current_grid = np.asarray(frame.grid, dtype=np.uint8).copy()

    for batch_index, batch in enumerate(recording.batches):
        if not _same_grid(current_grid, batch.start.grid):
            mismatch = AlignmentMismatch(
                batch_index=batch_index,
                event_index=batch.start.event_index,
                differing_cells=int(np.count_nonzero(current_grid != batch.start.grid)),
                expected_sha1=_digest(batch.start.grid),
                actual_sha1=_digest(current_grid),
            )
            mismatches.append(mismatch)
            if strict:
                raise ReplayAlignmentError(
                    f"{recording.recording_id}: pre-batch frame mismatch at batch {batch_index}"
                )

        for action, x, y in batch.actions:
            before = current_grid.copy()
            level_before = int(env.level)
            step = env.step(action, x, y)
            current_grid = np.asarray(step.frame.grid, dtype=np.uint8).copy()
            level_after = int(env.level)
            examples.append(
                DuckExample(
                    game_id=recording.game_code,
                    message_index=batch.start.event_index,
                    grid=before,
                    action=int(action),
                    x=x,
                    y=y,
                    reasoning=batch.reasoning,
                    source=recording.recording_id,
                    next_grid=current_grid.copy(),
                    visual_changed=not np.array_equal(before, current_grid),
                    level_advanced=bool(step.level_won or level_after > level_before),
                    terminal_result=str(env.state),
                    seed=seed,
                    trajectory_id=recording.recording_id,
                    level_before=level_before,
                    level_after=level_after,
                    action_index=action_index,
                )
            )
            action_index += 1

        if batch.expected is not None:
            if _same_grid(current_grid, batch.expected.grid):
                verified_anchors += 1
            else:
                mismatch = AlignmentMismatch(
                    batch_index=batch_index,
                    event_index=batch.expected.event_index,
                    differing_cells=int(np.count_nonzero(current_grid != batch.expected.grid)),
                    expected_sha1=_digest(batch.expected.grid),
                    actual_sha1=_digest(current_grid),
                )
                mismatches.append(mismatch)
                if strict:
                    raise ReplayAlignmentError(
                        f"{recording.recording_id}: frame mismatch after batch {batch_index}; "
                        f"expected={mismatch.expected_sha1} actual={mismatch.actual_sha1}"
                    )

    summary_success = _summary_success(recording.final_summary)
    success = bool(env.state == "WIN") if summary_success is None else summary_success
    total = len(examples)
    examples = [
        replace(
            example,
            remaining_path_length=total - index - 1,
            success=success,
        )
        for index, example in enumerate(examples)
    ]
    report = ReplayReport(
        recording_id=recording.recording_id,
        game_code=recording.game_code,
        game_id=recording.game_id,
        environment=environment_name,
        seed=seed,
        actions=total,
        batches=len(recording.batches),
        verified_anchors=verified_anchors,
        mismatches=tuple(mismatches),
        final_state=str(env.state),
        success=success,
    )
    return examples, report
