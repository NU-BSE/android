# Android DRLQN adapter

This adapter preserves the ARC-AGI-3 DRLQN implementation and adds a separate Android observation, planning, grounding, gesture and verification path.

`Differential Residual λ Q-Network` is the project architecture used by this repository, not a widely standardized paper title. The Android path combines recurrent Q-learning for partial observability, Peng Q(λ) targets, residual state updates and a small residual Bellman-gradient loss component.

## State contract

Every state is anchored to the screenshot timestamp:

```text
s_t = {
  screenshot_64x64,
  accessibility_tree,
  captions_or_teacher_OCR,
  package_and_activity,
  viewport_metadata,
  previous_actions,
  recent_accessibility_events
}
```

`TimestampSynchronizer` rejects accessibility trees outside the configured skew and drops stale captions/events. Keep all device records in one millisecond clock domain. Never join modalities by file order.

`normalize_screenshot` uses aspect-preserving letterboxing to 64×64 and returns the inverse transform for mapping heatmap points to device coordinates.

## Architecture

`drlqn/model/android_drlqn.py` provides:

- small 8×8-patch vision encoder;
- separate accessibility-tree encoder;
- text encoder for instruction, OCR/captions and UI labels;
- recent accessibility-event encoder;
- fusion transformer per frame;
- gated temporal encoder over 3–8 states;
- restricted semantic planner;
- node pointer, box head and heatmap grounder;
- gesture type, finger count, control points, duration and velocity heads;
- post-action verifier and next-state/world-model head;
- action-conditioned residual `lambda_update`.

The planner does not emit coordinates. Its grammar includes:

```text
OPEN_MENU
FIND_SETTING("Bluetooth")
ENTER_VALUE("...")
CONFIRM
```

The grounder selects a node, bounding box or heatmap cell. The gesture decoder then generates tap/swipe/drag/multi-touch parameters. Two to four control points per finger are sufficient for most Android UI trajectories.

## BFS window planner

`WindowGraph` creates state nodes from package, activity, active window and an accessibility-tree fingerprint. Demonstrated transitions become semantic edges. `shortest_plan` runs BFS over sufficiently reliable edges. Raw gesture coordinates stay attached only as demonstrations for grounding and decoding.

Use BFS for known window paths and the neural planner for unseen windows, dynamic content and recovery.

## Verification

`verify_transition` checks tree change, expected text, package/activity/window changes, target-node disappearance and whether the latest accessibility event indicates that the UI is still changing. A successful gesture callback alone is never considered task success.

## Collection

Run:

```bash
python scripts/collect_android.py --output data/android/demo-001 --episode-id demo-001
```

An Android `AccessibilityService` should emit timestamped JSON records under logcat tag `DRLQN_TRACE`. The host collector stores screenshot PNG, compressed UIAutomator XML, package/activity, viewport metadata and all timestamps.

UIAutomator is a bootstrap path. Production collection should serialize the full `AccessibilityNodeInfo` tree directly from the service so stable node IDs and event sources are preserved.

## Training curriculum

1. GUI grounding pretraining: text → node/box/heatmap.
2. Behavior cloning on human trajectories.
3. Curriculum: semantic node actions → tap/swipe → drag/slider → multi-touch → long tasks.
4. Hard negatives: duplicate labels, tiny icons, overlays and disabled state.
5. DAgger or human correction for diverged trajectories.
6. Offline Q(λ) for recovery and shorter paths.
7. Teacher distillation of subgoals and hidden representations.
8. World-model loss for predicted UI changes.

Use a target-network semi-gradient objective plus a small residual term (`residual_mix` around 0.05–0.15). Avoid pure residual-gradient Q-learning as the default.

Relevant foundations include DRQN (Hausknecht & Stone), modern Peng Q(λ), SeeClick-style GUI grounding, Android in the Wild and AndroidControl.
