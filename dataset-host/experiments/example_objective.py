"""Replace this inexpensive smoke-test objective with your actual model training."""

import math
from collections import Counter

import optuna

from clients.python_client import ArcDatasetClient


def objective(trial: optuna.Trial, dataset: ArcDatasetClient) -> float:
    game_code = trial.suggest_categorical("game_code", [g["game_code"] for g in dataset.games()])
    max_recordings = trial.suggest_int("max_recordings", 1, 3)
    frame_stride = trial.suggest_int("frame_stride", 1, 16, log=True)

    action_counts: Counter[int] = Counter()
    for recording_index, recording in enumerate(dataset.recordings(game_code)):
        if recording_index >= max_recordings:
            break
        for event_index, event in enumerate(dataset.events(recording["id"])):
            if event_index % frame_stride:
                continue
            action = event.get("data", {}).get("action_input", {}).get("id")
            if isinstance(action, int):
                action_counts[action] += 1

    total = sum(action_counts.values())
    if total == 0:
        return 0.0
    # A complete, cheap integration test: maximize sampled action-distribution entropy.
    return -sum((count / total) * math.log(count / total) for count in action_counts.values())
