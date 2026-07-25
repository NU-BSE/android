# VLM + chat coordinator for evidence-backed Android settings

The coordinator joins four independently testable components:

1. **ORM catalog** resolves a setting only for the exact OS/release/API/device scope.
2. **DRLQN perception** ranks accessibility nodes and may expose a box/heatmap through `PerceptionDRLQN`; gesture heads remain filtered.
3. **Local VLM** receives the screenshot, one semantic target and a minimized list of visible UI elements. It returns strict JSON with one normalized bounding box.
4. **Local chat LLM** explains exactly one semantic action through `PlannerAdvisor` and never receives executable commands or raw gesture trajectories.

```text
user goal
   -> SettingsCatalog.resolve(target scope)
   -> EvidenceBackedWorkflowPlanner
   -> PerceptionDRLQN / HybridGrounder / Local VLM
   -> ProposedAction + confirmation policy
   -> Android host executes or user performs the action
   -> ActionReceipt(before, after)
   -> verify_transition
   -> next chat turn
```

## ORM input

`InMemorySettingsCatalog.from_normalized_claims()` accepts the normalized extraction contract produced by the documentation parser. It does not crawl pages and does not execute document content.

Required properties for a usable capability:

- immutable `source_id` and `snapshot_sha256`;
- evidence confidence Tier 0-2;
- explicit OS family and compatible Android/API scope;
- documented `ui_path` for UI workflows;
- control plane, role, risk and reversibility metadata.

A setting with a mismatched OS family, unsupported device codename, stale/low-tier evidence or ambiguous name returns `None`. The coordinator reports the task as blocked instead of falling back to another ROM.

## Local VLM adapter

Implement `MultimodalGenerator` for the selected Android runtime, then wrap it with `JsonGroundingVLM`:

```python
vlm = JsonGroundingVLM(litert_multimodal_generator)
```

The generator receives:

- one screenshot;
- one target description;
- package/activity and viewport metadata;
- a minimized, redacted list of accessibility candidates.

It must return only:

```json
{
  "label": "Privacy dashboard",
  "role": "switch",
  "bbox_norm": [0.10, 0.20, 0.90, 0.30],
  "confidence": 0.91,
  "explanation": "The label and switch are visible in the privacy section",
  "metadata": {"model": "local-vlm"}
}
```

Unknown fields, invalid boxes, low confidence and executable text are rejected. Accessibility matching remains the first choice; DRLQN ranking is second; the VLM is the fallback.

## Chat LLM adapter

The existing `LocalChatModel` protocol is used by `PlannerAdvisor`. The LLM receives only the current semantic subgoal and redacted visible labels. It explains one next action and asks for confirmation. It does not decide whether a setting exists or whether the device is in scope; those decisions come from the ORM catalog and policy code.

## Action boundary

`SettingsCoordinator` returns a `ProposedAction`. The repository still does not provide an AccessibilityService executor. A host application may:

- show the instruction and let the user perform it manually; or
- pass the proposal to a separately reviewed executor after confirmation.

The host must return an `ActionReceipt` containing the exact `action_id`, before/after states, success status and user confirmation. The coordinator advances only after `verify_transition()` confirms the expected UI change.

Medium, high and critical-risk final changes require explicit confirmation. Public APIs, DevicePolicyManager, Android Management API, shell, Settings Provider, Shizuku and root require separate approved executors; the built-in planner currently creates UI/manual workflows only.

## Android integration

A typical Android host should maintain these interfaces:

```text
AccessibilityService / MediaProjection
    -> timestamp-aligned AndroidState
LiteRT-LM or ONNX Runtime multimodal adapter
    -> LocalVisionLanguageModel
LiteRT-LM text adapter
    -> LocalChatModel
Room/SQL ORM repository
    -> SettingsCatalog
UI / policy layer
    -> confirmation + ActionReceipt
```

Do not upload full build fingerprints or unredacted UI text unless required and disclosed. Keep model context local where possible and persist only evidence IDs, workflow IDs, receipts and redacted audit metadata.
