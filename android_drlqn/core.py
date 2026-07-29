"""Timestamp-aligned Android observations for the DRLQN agent."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

ANDROID_IMAGE_SIZE = 64
ACCESS_NODE_DIM = 24
TEXT_TOKEN_DIM = 64
META_DIM = 16
ACTION_HISTORY_DIM = 16
EVENT_DIM = 24


@dataclass(frozen=True)
class Bounds:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def center(self):
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)


@dataclass(frozen=True)
class ViewportMetadata:
    width_px: int
    height_px: int
    density: float = 1.0
    rotation: int = 0
    status_bar_px: int = 0
    navigation_bar_px: int = 0
    ime_height_px: int = 0


@dataclass(frozen=True)
class AccessibilityNode:
    node_id: str
    parent_id: str | None = None
    class_name: str = ""
    package: str = ""
    resource_id: str = ""
    text: str = ""
    content_description: str = ""
    bounds: Bounds = Bounds(0, 0, 0, 0)
    window_id: int = 0
    enabled: bool = True
    clickable: bool = False
    scrollable: bool = False
    editable: bool = False
    checked: bool = False
    selected: bool = False
    focused: bool = False
    visible: bool = True

    @property
    def label(self):
        return self.text or self.content_description or self.resource_id or self.class_name


@dataclass(frozen=True)
class AccessibilitySnapshot:
    timestamp_ms: int
    nodes: tuple[AccessibilityNode, ...]
    active_window_id: int = 0


@dataclass(frozen=True)
class ScreenshotFrame:
    timestamp_ms: int
    pixels: np.ndarray


@dataclass(frozen=True)
class Caption:
    timestamp_ms: int
    text: str
    source: str = "ocr"
    confidence: float = 1.0


@dataclass(frozen=True)
class UIEvent:
    timestamp_ms: int
    event_type: str
    node_id: str | None = None
    text: str = ""
    window_id: int = 0
    package: str = ""
    activity: str = ""


class GestureType(str, Enum):
    TAP = "tap"
    LONG_PRESS = "long_press"
    SWIPE = "swipe"
    DRAG = "drag"
    TYPE = "type"
    BACK = "back"
    HOME = "home"
    MULTI_TOUCH = "multi_touch"
    WAIT = "wait"


@dataclass(frozen=True)
class GestureAction:
    gesture_type: GestureType
    timestamp_ms: int
    points: tuple[tuple[float, float], ...] = ()
    duration_ms: int = 0
    number_of_fingers: int = 1
    text: str = ""
    target_node_id: str | None = None


@dataclass(frozen=True)
class AndroidState:
    screenshot: ScreenshotFrame
    accessibility: AccessibilitySnapshot
    captions: tuple[Caption, ...]
    package: str
    activity: str
    viewport: ViewportMetadata
    previous_actions: tuple[GestureAction, ...] = ()
    recent_events: tuple[UIEvent, ...] = ()


@dataclass(frozen=True)
class SynchronizedState:
    state: AndroidState
    tree_skew_ms: int
    caption_skews_ms: tuple[int, ...]
    event_ages_ms: tuple[int, ...]


class AlignmentError(ValueError):
    pass


class TimestampSynchronizer:
    def __init__(self, max_tree_skew_ms=180, max_caption_skew_ms=300,
                 event_lookback_ms=1200, max_future_event_ms=40):
        self.max_tree_skew_ms = int(max_tree_skew_ms)
        self.max_caption_skew_ms = int(max_caption_skew_ms)
        self.event_lookback_ms = int(event_lookback_ms)
        self.max_future_event_ms = int(max_future_event_ms)

    def synchronize(self, screenshot, trees, captions, package, activity, viewport,
                    previous_actions=(), events=()):
        if not trees:
            raise AlignmentError("accessibility snapshot is required")
        anchor = int(screenshot.timestamp_ms)
        tree = min(trees, key=lambda x: abs(int(x.timestamp_ms) - anchor))
        skew = int(tree.timestamp_ms) - anchor
        if abs(skew) > self.max_tree_skew_ms:
            raise AlignmentError(f"accessibility tree skew {skew}ms exceeds limit")
        captions = tuple(sorted((c for c in captions if abs(c.timestamp_ms-anchor) <= self.max_caption_skew_ms), key=lambda x:x.timestamp_ms))
        events = tuple(sorted((e for e in events if -self.max_future_event_ms <= anchor-e.timestamp_ms <= self.event_lookback_ms), key=lambda x:x.timestamp_ms))
        state = AndroidState(screenshot, tree, captions, package, activity, viewport,
                             tuple(previous_actions), events)
        return SynchronizedState(state, skew,
            tuple(c.timestamp_ms-anchor for c in captions),
            tuple(anchor-e.timestamp_ms for e in events))


@dataclass(frozen=True)
class SquareTransform:
    source_width: int
    source_height: int
    scale: float
    pad_x: int
    pad_y: int
    size: int = ANDROID_IMAGE_SIZE

    def point_to_device(self, x, y):
        return ((x-self.pad_x)/self.scale, (y-self.pad_y)/self.scale)


def normalize_screenshot(pixels, size=ANDROID_IMAGE_SIZE):
    image = np.asarray(pixels)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, -1)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError("screenshot must be HxW[x3|x4]")
    image = image[..., :3]
    h, w = image.shape[:2]
    scale = min(size/w, size/h)
    rw, rh = max(1, round(w*scale)), max(1, round(h*scale))
    xs = np.minimum(w-1, (np.arange(rw)*w/rw).astype(int))
    ys = np.minimum(h-1, (np.arange(rh)*h/rh).astype(int))
    resized = image[ys[:, None], xs[None, :]].astype(float)
    if np.issubdtype(image.dtype, np.integer):
        resized /= np.iinfo(image.dtype).max
    elif resized.max(initial=0) > 1:
        resized /= 255
    px, py = (size-rw)//2, (size-rh)//2
    out = np.zeros((size, size, 3), float)
    out[py:py+rh, px:px+rw] = resized
    return np.clip(out, 0, 1), SquareTransform(w, h, scale, px, py, size)


def _hash(text, dim):
    out = np.zeros(dim)
    for token in (text or "").lower().split():
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        out[value % dim] += 1 if value & 256 else -1
    norm = np.linalg.norm(out)
    return out / norm if norm else out


def tree_fingerprint(snapshot):
    rows = [f"{n.node_id}|{n.parent_id}|{n.resource_id}|{n.label}|{n.bounds}|{n.enabled}|{n.visible}" for n in sorted(snapshot.nodes, key=lambda n:n.node_id)]
    return hashlib.sha1("\n".join(rows).encode()).hexdigest()[:20]


@dataclass(frozen=True)
class AndroidBatchConfig:
    history: int = 8
    max_nodes: int = 96
    max_text_tokens: int = 64
    max_events: int = 32
    image_size: int = 64

    def __post_init__(self):
        if not 3 <= self.history <= 8:
            raise ValueError("history must be in [3, 8]")


def prepare_model_inputs(history, instruction, config=AndroidBatchConfig()):
    states = [x.state if isinstance(x, SynchronizedState) else x for x in history][-config.history:]
    B, H = 1, config.history
    result = {
        "screenshots": np.zeros((B,H,config.image_size,config.image_size,3)),
        "node_features": np.zeros((B,H,config.max_nodes,ACCESS_NODE_DIM)),
        "node_mask": np.zeros((B,H,config.max_nodes), bool),
        "text_features": np.zeros((B,H,config.max_text_tokens,TEXT_TOKEN_DIM)),
        "text_mask": np.zeros((B,H,config.max_text_tokens), bool),
        "metadata": np.zeros((B,H,META_DIM)),
        "previous_actions": np.zeros((B,H,ACTION_HISTORY_DIM)),
        "events": np.zeros((B,H,config.max_events,EVENT_DIM)),
        "event_mask": np.zeros((B,H,config.max_events), bool),
        "history_mask": np.zeros((B,H), bool),
    }
    node_ids = [tuple() for _ in range(H)]
    for t, state in enumerate(states, start=H-len(states)):
        result["screenshots"][0,t] = normalize_screenshot(state.screenshot.pixels, config.image_size)[0]
        vw, vh = max(1,state.viewport.width_px), max(1,state.viewport.height_px)
        for i,node in enumerate(state.accessibility.nodes[:config.max_nodes]):
            b=node.bounds; f=result["node_features"][0,t,i]
            f[:8]=[b.left/vw,b.top/vh,b.right/vw,b.bottom/vh,node.enabled,node.clickable,node.scrollable,node.visible]
            f[8:]=_hash(" ".join((node.label,node.class_name,node.resource_id)), ACCESS_NODE_DIM-8)
            result["node_mask"][0,t,i]=True
        texts=[instruction]+[c.text for c in state.captions]+[n.label for n in state.accessibility.nodes if n.label]
        for i,text in enumerate(texts[:config.max_text_tokens]):
            result["text_features"][0,t,i]=_hash(text,TEXT_TOKEN_DIM); result["text_mask"][0,t,i]=True
        result["metadata"][0,t,:4]=[vw/max(vh,1),vh/max(vw,1),state.viewport.density,state.viewport.rotation/3]
        result["metadata"][0,t,4:]=_hash(state.package+" "+state.activity,META_DIM-4)
        for i,event in enumerate(state.recent_events[-config.max_events:]):
            result["events"][0,t,i,:8]=_hash(event.event_type,8)
            result["events"][0,t,i,8:]=_hash(event.text+event.activity,EVENT_DIM-8)
            result["event_mask"][0,t,i]=True
        result["history_mask"][0,t]=True
        node_ids[t]=tuple(n.node_id for n in state.accessibility.nodes[:config.max_nodes])
    result["node_ids"] = tuple(node_ids)
    return result


@dataclass(frozen=True)
class VerificationExpectation:
    expected_text: str | None = None
    expected_package: str | None = None
    expected_activity: str | None = None
    target_node_id: str | None = None
    require_tree_change: bool = False
    require_window_change: bool = False
    wait_for_idle: bool = True


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    tree_changed: bool
    expected_text_present: bool
    activity_changed: bool
    window_changed: bool
    target_disappeared: bool
    idle: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def verify_transition(before, after, expectation):
    tree_changed = tree_fingerprint(before.accessibility) != tree_fingerprint(after.accessibility)
    text = " ".join([c.text for c in after.captions]+[n.label for n in after.accessibility.nodes]).lower()
    text_ok = not expectation.expected_text or expectation.expected_text.lower() in text
    activity_changed = (before.package,before.activity)!=(after.package,after.activity)
    window_changed = activity_changed or before.accessibility.active_window_id != after.accessibility.active_window_id
    target_gone = not expectation.target_node_id or expectation.target_node_id not in {n.node_id for n in after.accessibility.nodes}
    latest=max((e.timestamp_ms for e in after.recent_events),default=-1)
    idle=not any(e.timestamp_ms==latest and e.event_type in ("TYPE_WINDOW_CONTENT_CHANGED","TYPE_WINDOWS_CHANGED") for e in after.recent_events)
    reasons=[]
    if expectation.require_tree_change and not tree_changed: reasons.append("tree unchanged")
    if not text_ok: reasons.append("expected text absent")
    if expectation.expected_package and after.package!=expectation.expected_package: reasons.append("package mismatch")
    if expectation.expected_activity and after.activity!=expectation.expected_activity: reasons.append("activity mismatch")
    if expectation.require_window_change and not window_changed: reasons.append("window unchanged")
    if not target_gone: reasons.append("target still present")
    if expectation.wait_for_idle and not idle: reasons.append("UI still changing")
    return VerificationResult(not reasons,tree_changed,text_ok,activity_changed,window_changed,target_gone,idle,tuple(reasons))
