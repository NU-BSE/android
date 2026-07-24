# Android DRLQN

Minimal Android-only implementation of the Differential Residual Lambda Q-Network adaptation.

The branch intentionally excludes ARC-AGI environments, games, operators, API adapters, result artifacts and ARC training scripts.

## Included

- timestamp-aligned screenshot, accessibility, OCR/caption and UI-event state;
- 64×64 aspect-preserving screenshot normalization;
- semantic subgoal grammar and observed-window BFS planner;
- compact multimodal model with vision, accessibility, text, event, temporal and fusion encoders;
- pointer/box/heatmap grounding, gesture decoding, transition verification and world-model prediction;
- Peng Q(λ), residual-Q and multitask losses;
- ADB/logcat trajectory collector.

## Install

```bash
python -m pip install -e '.[test]'
```

## Test

```bash
pytest
```

## Collect a trajectory

An Android `AccessibilityService` should emit JSON events to the `DRLQN_TRACE` logcat tag with device timestamps.

```bash
python scripts/collect_android.py --output data/session-001 --serial DEVICE_SERIAL
```

Training data must be joined by timestamps, not by file order. Stale accessibility trees and captions are rejected by `TimestampSynchronizer`.
