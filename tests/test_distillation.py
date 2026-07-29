"""On-policy distillation, Ollama teacher adapters and emulator profiles."""
from __future__ import annotations

import io
import json
import struct
import zlib

import numpy as np
import pytest

from android_drlqn.core import (
    AccessibilityNode,
    AccessibilitySnapshot,
    AndroidBatchConfig,
    AndroidState,
    Bounds,
    ScreenshotFrame,
    ViewportMetadata,
)
from android_drlqn.distillation import (
    DistillationConfig,
    DistillationTask,
    OnPolicyDistiller,
    box_iou,
    build_student,
    load_checkpoint,
    node_boxes,
    run_sequential_distillation,
    save_checkpoint,
)
from android_drlqn.emulator import (
    AdbClient,
    AdbResearchActuator,
    EmulatorError,
    EmulatorFleet,
    EmulatorProfile,
    decode_screencap,
    load_profiles,
    parse_ui_dump,
)
from android_drlqn.ollama import (
    GROUNDING_SCHEMA,
    OllamaClient,
    OllamaConfig,
    OllamaMultimodalGenerator,
    downscale,
    encode_png,
    normalize_grounding_payload,
    ollama_grounding_vlm,
)
from android_drlqn.vlm import (
    GrounderConfig,
    GroundingSource,
    HybridGrounder,
    NormalizedBox,
)

WIDTH, HEIGHT = 1000, 2000
BATCH_CONFIG = AndroidBatchConfig(history=3, max_nodes=8)


def _state(labels=("Connected devices", "Storage"), *, offset: int = 0) -> AndroidState:
    nodes = tuple(
        AccessibilityNode(
            node_id=f"0.{index}",
            text=label,
            class_name="android.widget.TextView",
            bounds=Bounds(0, 200 * (index + offset), WIDTH, 200 * (index + offset) + 180),
            clickable=True,
        )
        for index, label in enumerate(labels)
    )
    return AndroidState(
        screenshot=ScreenshotFrame(1_000, np.zeros((64, 64, 3), dtype=np.uint8)),
        accessibility=AccessibilitySnapshot(1_000, nodes, active_window_id=1 + offset),
        captions=(),
        package="com.android.settings",
        activity=".Settings",
        viewport=ViewportMetadata(WIDTH, HEIGHT),
    )


class ScriptedTeacher:
    """Always grounds the second row, so supervision is deterministic."""

    def __init__(self, confidence: float = 0.9):
        self.confidence = confidence
        self.calls = 0

    def ground(self, request):
        self.calls += 1
        return json.dumps(
            {
                "label": "Storage",
                "role": "android.widget.TextView",
                "bbox_norm": [0.0, 200 / HEIGHT, 1.0, 380 / HEIGHT],
                "confidence": self.confidence,
                "explanation": "second visible row",
                "metadata": {"model": "scripted"},
            }
        )


class FakeEnvironment:
    def __init__(self, screens):
        self.screens = list(screens)
        self.index = 0
        self.taps: list[NormalizedBox] = []

    def reset(self):
        self.index = 0
        return self.screens[0]

    def capture(self):
        return self.screens[min(self.index, len(self.screens) - 1)]

    def act(self, box):
        self.taps.append(box)
        self.index = min(self.index + 1, len(self.screens) - 1)
        return self.capture()

    def back(self):
        self.index = max(self.index - 1, 0)
        return self.capture()

    def scroll(self):
        return self.capture()


def _config(**overrides) -> DistillationConfig:
    base = {
        "batch_size": 1,
        "max_steps_per_profile": 6,
        "exploration": 0.0,
        "batch_config": BATCH_CONFIG,
        "learning_rate": 1e-3,
    }
    base.update(overrides)
    return DistillationConfig(**base)


# --------------------------------------------------------------------- teacher
def test_teacher_supervision_maps_box_onto_accessibility_nodes():
    config = _config()
    distiller = OnPolicyDistiller(build_student(config), ScriptedTeacher(), config)
    supervision, reason = distiller.teacher_supervision(_state(), "Storage")

    assert reason == ""
    assert supervision is not None
    assert supervision.node_index == 1
    assert pytest.approx(supervision.node_targets.sum()) == 1.0
    assert supervision.node_targets.shape == (BATCH_CONFIG.max_nodes,)


def test_low_confidence_teacher_answer_is_rejected():
    config = _config(min_teacher_confidence=0.95, teacher_vlm_only=True)
    distiller = OnPolicyDistiller(build_student(config), ScriptedTeacher(0.7), config)
    supervision, reason = distiller.teacher_supervision(_state(), "Storage")

    assert supervision is None
    assert "confidence" in reason


def test_teacher_box_without_node_overlap_is_rejected():
    class OffscreenTeacher:
        def ground(self, request):
            return json.dumps(
                {
                    "label": "ghost",
                    "role": "view",
                    "bbox_norm": [0.0, 0.90, 0.1, 0.99],
                    "confidence": 0.99,
                    "explanation": "nowhere near a node",
                    "metadata": {},
                }
            )

    config = _config(teacher_vlm_only=True)
    distiller = OnPolicyDistiller(build_student(config), OffscreenTeacher(), config)
    supervision, reason = distiller.teacher_supervision(_state(), "Storage")

    assert supervision is None
    assert "overlap" in reason


def test_vlm_only_teacher_skips_the_accessibility_shortcut():
    teacher = ScriptedTeacher()
    config = _config(teacher_vlm_only=True)
    distiller = OnPolicyDistiller(build_student(config), teacher, config)
    supervision, _ = distiller.teacher_supervision(_state(), "Storage")

    assert teacher.calls == 1
    assert supervision is not None
    assert supervision.source is GroundingSource.VLM


def test_default_teacher_prefers_the_accessibility_tree():
    teacher = ScriptedTeacher()
    config = _config()
    distiller = OnPolicyDistiller(build_student(config), teacher, config)
    supervision, _ = distiller.teacher_supervision(_state(), "Storage")

    assert supervision is not None
    assert supervision.source is GroundingSource.ACCESSIBILITY


# ---------------------------------------------------------------------- update
def test_distillation_update_moves_student_towards_the_teacher_node():
    config = _config()
    student = build_student(config, seed=3)
    distiller = OnPolicyDistiller(student, ScriptedTeacher(), config)
    state = _state()
    history = [state]

    supervision, _ = distiller.teacher_supervision(state, "Storage")
    before = distiller.student_outputs(history, "goal")
    start = float(before["ground_node_logits"].data[0, supervision.node_index])

    for _ in range(15):
        outputs = distiller.student_outputs(history, "goal")
        distiller.accumulate(outputs, supervision)
        distiller.flush(force=True)

    after = distiller.student_outputs(history, "goal")
    end = float(after["ground_node_logits"].data[0, supervision.node_index])
    probability = float(after["ground_node_logits"].softmax(axis=-1).data[0, supervision.node_index])

    assert end > start
    assert probability > 0.5


def test_gesture_heads_never_receive_distillation_gradients():
    config = _config()
    student = build_student(config)
    distiller = OnPolicyDistiller(student, ScriptedTeacher(), config)
    state = _state()
    supervision, _ = distiller.teacher_supervision(state, "Storage")

    distiller.accumulate(distiller.student_outputs([state], "goal"), supervision)
    distiller.optimizer.zero_grad()
    total = distiller._pending[0][0] + distiller._pending[0][1]
    total.backward()

    gesture_parameters = (
        student.control_points.parameters()
        + student.duration.parameters()
        + student.velocity.parameters()
        + student.gesture_type.parameters()
        + student.finger_count.parameters()
    )
    assert all(parameter.grad is None for parameter in gesture_parameters)


def test_batching_defers_the_update_until_the_batch_is_full():
    config = _config(batch_size=3)
    distiller = OnPolicyDistiller(build_student(config), ScriptedTeacher(), config)
    state = _state()
    supervision, _ = distiller.teacher_supervision(state, "Storage")

    assert distiller.flush() is False
    for expected in (False, False, True):
        distiller.accumulate(distiller.student_outputs([state], "goal"), supervision)
        assert distiller.flush() is expected


# --------------------------------------------------------------------- rollout
def test_sequential_run_visits_every_profile_and_keeps_one_student():
    config = _config(max_steps_per_profile=4)
    student = build_student(config)
    task = DistillationTask("Открыть хранилище", ("Storage",), max_attempts_per_target=2)
    profiles = (
        EmulatorProfile("p1", "avd1", 12, serial="emulator-5554"),
        EmulatorProfile("p2", "avd2", 15, serial="emulator-5556"),
    )
    environments = [
        (profile, FakeEnvironment([_state(), _state(offset=1)])) for profile in profiles
    ]

    report = run_sequential_distillation(student, ScriptedTeacher(), environments, (task,), config)

    assert [summary.profile for summary in report.profiles] == ["p1", "p2"]
    assert all(summary.steps > 0 for summary in report.profiles)
    assert all(summary.accepted > 0 for summary in report.profiles)
    assert report.total_updates > 0
    assert report.to_dict()["profiles"][0]["android_version"] == 12


def test_rollout_taps_the_node_the_student_chose_not_the_teacher_box():
    config = _config(max_steps_per_profile=1, exploration=0.0)
    distiller = OnPolicyDistiller(build_student(config, seed=7), ScriptedTeacher(), config)
    environment = FakeEnvironment([_state(), _state(offset=1)])
    profile = EmulatorProfile("p1", "avd1", 15, serial="emulator-5554")

    summary = distiller.run_profile(
        profile, environment, (DistillationTask("goal", ("Storage",)),)
    )

    assert summary.steps == 1
    assert len(environment.taps) == 1
    boxes = node_boxes(_state(), BATCH_CONFIG.max_nodes)
    assert environment.taps[0] in boxes


def test_report_serializes_rejections_per_profile(tmp_path):
    config = _config(min_teacher_confidence=0.99, max_steps_per_profile=2, teacher_vlm_only=True)
    student = build_student(config)
    environments = [
        (
            EmulatorProfile("p1", "avd1", 15, serial="emulator-5554"),
            FakeEnvironment([_state()]),
        )
    ]
    report = run_sequential_distillation(
        student, ScriptedTeacher(0.5), environments, (DistillationTask("g", ("Storage",)),), config
    )
    path = tmp_path / "report.json"
    report.write(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["profiles"][0]["accepted"] == 0
    assert sum(payload["profiles"][0]["rejected"].values()) > 0
    assert payload["records"]


def test_checkpoint_round_trip(tmp_path):
    config = _config()
    first = build_student(config, seed=1)
    second = build_student(config, seed=2)
    path = tmp_path / "student.npz"

    save_checkpoint(first, path)
    load_checkpoint(second, path)

    for left, right in zip(first.state_dict(), second.state_dict()):
        assert np.allclose(left, right)


def test_box_iou_and_node_box_ordering():
    boxes = node_boxes(_state(), BATCH_CONFIG.max_nodes)
    assert len(boxes) == 2
    assert box_iou(boxes[0], boxes[0]) == pytest.approx(1.0)
    assert box_iou(boxes[0], boxes[1]) == 0.0


# ----------------------------------------------------------------------- ollama
def test_encode_png_is_readable_and_preserves_size():
    pixels = np.zeros((5, 7, 3), dtype=np.uint8)
    pixels[2, 3] = (255, 128, 64)
    payload = encode_png(pixels)

    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, depth, colour = struct.unpack(">IIBB", payload[16:26])
    assert (width, height, depth, colour) == (7, 5, 8, 2)

    stream = io.BytesIO(payload[8:])
    raw = b""
    while chunk_header := stream.read(8):
        length, tag = struct.unpack(">I", chunk_header[:4])[0], chunk_header[4:]
        body = stream.read(length)
        stream.read(4)
        if tag == b"IDAT":
            raw += body
    decoded = zlib.decompress(raw)
    assert len(decoded) == height * (width * 3 + 1)
    assert decoded[0] == 0  # filter byte
    row = decoded[2 * (width * 3 + 1) + 1 :][: width * 3]
    assert tuple(row[9:12]) == (255, 128, 64)


def test_encode_png_normalizes_float_images():
    payload = encode_png(np.ones((2, 2, 3), dtype=np.float32))
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"


def test_downscale_caps_the_longest_side():
    small = downscale(np.zeros((2400, 1080, 3), dtype=np.uint8), 512)
    assert max(small.shape[:2]) == 512
    unchanged = np.zeros((10, 10, 3), dtype=np.uint8)
    assert downscale(unchanged, 512) is unchanged


class FakeHTTP:
    """Minimal urlopen stand-in that records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        payload = json.dumps(self.responses.pop(0)).encode()

        class Response(io.BytesIO):
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *args):
                return False

        return Response(payload)


def test_ollama_generator_attaches_the_image_to_the_user_message():
    http = FakeHTTP([{"message": {"content": "{}"}}])
    config = OllamaConfig(model="qwen2.5vl:7b")
    generator = OllamaMultimodalGenerator(config, client=OllamaClient(config, opener=http))

    generator.generate(
        np.zeros((4, 4, 3), dtype=np.uint8),
        ({"role": "system", "content": "s"}, {"role": "user", "content": "u"}),
        max_new_tokens=32,
    )

    body = json.loads(http.requests[0].data)
    assert body["model"] == "qwen2.5vl:7b"
    assert body["format"] == "json"
    assert body["stream"] is False
    assert body["options"]["num_predict"] == 32
    assert "images" not in body["messages"][0]
    assert len(body["messages"][1]["images"]) == 1


def test_grounding_requests_use_a_json_schema_not_plain_json_mode():
    http = FakeHTTP([{"message": {"content": "{}"}}])
    config = OllamaConfig(model="qwen2.5vl:7b")
    ollama_grounding_vlm(config, client=OllamaClient(config, opener=http), verify_installed=False)
    generator = OllamaMultimodalGenerator(
        config, client=OllamaClient(config, opener=http), response_schema=GROUNDING_SCHEMA
    )
    generator.generate(
        np.zeros((4, 4, 3), dtype=np.uint8),
        ({"role": "user", "content": "u"},),
        max_new_tokens=256,
    )

    body = json.loads(http.requests[0].data)
    assert body["format"]["required"] == [
        "label",
        "role",
        "bbox_norm",
        "confidence",
        "explanation",
    ]
    # Without maxLength a verbose teacher runs past num_predict and the object
    # is never closed, which the contract then rejects as malformed.
    assert body["format"]["properties"]["explanation"]["maxLength"] > 0


def test_dialect_repair_rescales_qwen_style_0_1000_boxes():
    payload = normalize_grounding_payload(
        json.dumps(
            {
                "label": "Storage",
                "role": "row",
                "bbox_norm": [50, 375, 950, 525],
                "confidence": 0.9,
                "explanation": "row",
            }
        )
    )
    assert payload["bbox_norm"] == [0.05, 0.375, 0.95, 0.525]
    assert HybridGrounder.parse_vlm_response(payload).box.top == 0.375


def test_dialect_repair_drops_stray_keys_and_fences():
    payload = normalize_grounding_payload(
        '```json\n{"label":"Storage","role":"row","bbox_norm":[0.0,0.1,1.0,0.2],'
        '"confidence":0.8,"explanation":"row"," ":"confidence","thoughts":"..."}\n```'
    )
    assert set(payload) == {"label", "role", "bbox_norm", "confidence", "explanation"}
    assert HybridGrounder.parse_vlm_response(payload).confidence == 0.8


def test_dialect_repair_still_refuses_executable_teacher_text():
    payload = normalize_grounding_payload(
        json.dumps(
            {
                "label": "adb shell input tap 10 20",
                "role": "row",
                "bbox_norm": [0.0, 0.1, 1.0, 0.2],
                "confidence": 0.9,
                "explanation": "x",
            }
        )
    )
    with pytest.raises(ValueError):
        HybridGrounder.parse_vlm_response(payload)


def test_vlm_only_teacher_counts_a_contract_violation_instead_of_falling_back():
    class BrokenTeacher:
        def ground(self, request):
            return "not json at all"

    config = _config(teacher_vlm_only=True)
    distiller = OnPolicyDistiller(build_student(config), BrokenTeacher(), config)
    supervision, reason = distiller.teacher_supervision(_state(), "Storage")

    assert supervision is None
    assert "contract" in reason


def test_grounder_config_rejects_a_truncating_token_budget():
    with pytest.raises(ValueError):
        GrounderConfig(vlm_max_new_tokens=96)


def test_build_request_passes_the_configured_token_budget():
    grounder = HybridGrounder(GrounderConfig(vlm_max_new_tokens=320))
    request = grounder.build_request(_state(), "Storage")

    assert request.max_new_tokens == 320
    assert [element.label for element in request.visible_elements] == [
        "Connected devices",
        "Storage",
    ]


def test_ensure_model_reports_a_missing_tag():
    config = OllamaConfig(model="qwen2.5vl:7b")
    client = OllamaClient(config, opener=FakeHTTP([{"models": [{"name": "qwen2.5vl:32b"}]}]))

    with pytest.raises(Exception) as error:
        client.ensure_model()
    assert "ollama pull" in str(error.value)


# --------------------------------------------------------------------- emulator
def test_profile_scope_matches_the_api_table():
    profile = EmulatorProfile("p", "avd", 16, os_family="grapheneos", os_release="2026.07")
    scope = profile.target_scope()

    assert profile.api_level == 36
    assert scope.api_level == 36
    assert scope.os_family == "grapheneos"
    assert profile.device_capabilities().android_version == 16


def test_profile_rejects_a_non_emulator_serial():
    with pytest.raises(ValueError):
        EmulatorProfile("p", "avd", 15, serial="R58N1234ABC")


def test_load_profiles_keeps_order_and_rejects_unknown_fields(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {"name": "b", "avd_name": "avd_b", "android_version": 15},
                    {"name": "a", "avd_name": "avd_a", "android_version": 12},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert [profile.name for profile in load_profiles(path)] == ["b", "a"]

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"profiles": [{"name": "x", "avd_name": "a", "android_version": 15, "gpu": 1}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_profiles(bad)


def test_repository_profile_config_loads():
    profiles = load_profiles("config/emulator_profiles.json")
    assert len(profiles) >= 2
    assert all(profile.serial.startswith("emulator-") for profile in profiles)


def test_parse_ui_dump_builds_stable_node_paths():
    xml = """<?xml version='1.0'?>
    <hierarchy rotation="0">
      <node class="android.widget.FrameLayout" package="com.android.settings"
            bounds="[0,0][1080,2400]" enabled="true" clickable="false">
        <node class="android.widget.TextView" text="Storage" package="com.android.settings"
              resource-id="android:id/title" bounds="[0,200][1080,380]"
              enabled="true" clickable="true" checked="false"/>
      </node>
    </hierarchy>"""
    nodes = parse_ui_dump(xml)

    assert [node.node_id for node in nodes] == ["0", "0.0"]
    assert nodes[1].parent_id == "0"
    assert nodes[1].text == "Storage"
    assert nodes[1].clickable and nodes[1].visible
    assert nodes[1].bounds.top == 200


def test_decode_screencap_handles_both_header_layouts():
    width, height = 3, 2
    pixels = np.arange(width * height * 4, dtype=np.uint8).reshape(height, width, 4)
    for header in (
        struct.pack("<III", width, height, 1),
        struct.pack("<IIII", width, height, 1, 0),
    ):
        decoded = decode_screencap(header + pixels.tobytes())
        assert decoded.shape == (height, width, 3)
        assert np.array_equal(decoded, pixels[..., :3])


def test_decode_screencap_rejects_a_truncated_payload():
    with pytest.raises(EmulatorError):
        decode_screencap(struct.pack("<III", 4, 4, 1) + b"\x00" * 10)


def _fake_adb(properties):
    def runner(command, binary):
        text = " ".join(command)
        if "devices" in command:
            return b"List of devices attached\nemulator-5554\tdevice\n"
        for key, value in properties.items():
            if key in text:
                return value.encode()
        return b""

    return runner


def test_actuator_refuses_without_explicit_opt_in():
    adb = AdbClient("emulator-5554", runner=_fake_adb({"ro.kernel.qemu": "1"}))
    with pytest.raises(EmulatorError):
        AdbResearchActuator(adb, ViewportMetadata(1080, 2400), allow_input=False)


def test_actuator_refuses_a_physical_device():
    def runner(command, binary):
        if "devices" in command:
            return b"List of devices attached\nR58N1234ABC\tdevice\n"
        return b"0"

    adb = AdbClient("R58N1234ABC", runner=runner)
    with pytest.raises(EmulatorError):
        AdbResearchActuator(adb, ViewportMetadata(1080, 2400), allow_input=True)


def test_actuator_taps_the_box_centre_in_device_pixels():
    commands: list[list[str]] = []

    def runner(command, binary):
        commands.append(list(command))
        if "devices" in command:
            return b"List of devices attached\nemulator-5554\tdevice\n"
        if "ro.kernel.qemu" in " ".join(command):
            return b"1"
        return b""

    adb = AdbClient("emulator-5554", runner=runner)
    actuator = AdbResearchActuator(
        adb, ViewportMetadata(1000, 2000), allow_input=True, settle_ms=0
    )
    actuator.tap(NormalizedBox(0.0, 0.1, 1.0, 0.3))

    tap = [command for command in commands if "tap" in command][-1]
    assert tap[-2:] == ["500", "400"]


def test_fleet_boots_profiles_one_at_a_time():
    started: list[str] = []
    stopped: list[str] = []

    class FakeProcess:
        def __init__(self, name):
            self.name = name

        def terminate(self):
            stopped.append(self.name)

    def launcher(profile):
        started.append(profile.name)
        return FakeProcess(profile.name)

    def runner(command, binary):
        if "devices" in command:
            return b"List of devices attached\nemulator-5554\tdevice\nemulator-5556\tdevice\n"
        if "sys.boot_completed" in " ".join(command):
            return b"1"
        if "ro.kernel.qemu" in " ".join(command):
            return b"1"
        return b""

    profiles = (
        EmulatorProfile("p1", "avd1", 12, serial="emulator-5554"),
        EmulatorProfile("p2", "avd2", 15, serial="emulator-5556"),
    )
    fleet = EmulatorFleet(runner=runner, allow_input=True, boot_launcher=launcher)

    seen = []
    for profile, environment in fleet.sequential(profiles):
        # Both serials are already attached, so nothing needs launching.
        seen.append((profile.name, environment.profile.name))

    assert seen == [("p1", "p1"), ("p2", "p2")]
    assert started == []
    assert stopped == []
