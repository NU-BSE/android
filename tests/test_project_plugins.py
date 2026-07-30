"""Tests for the project_train / project_lab plugin modules."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from android_drlqn.distillation import (
    DistillationSample,
    EnvironmentOutcome,
    OnPolicyConfig,
    ProposalKind,
    RolloutObservation,
    StudentProposal,
    TeacherJudgement,
    TeacherTarget,
)
from android_drlqn.document_curriculum import CurriculumTask, CurriculumTaskKind
from android_drlqn.emulator_lab import selected_emulator_matrix
from android_drlqn.emulator_runtime import AdbClient, CommandResult

import project_lab
import project_train
from project_train import (
    ACTION_PREFIX,
    PackedExample,
    build_messages,
    canonical_action_json,
    pack_examples,
    parse_action_json,
    sanitize_bbox,
    speedrun_lr,
    teacher_completion_text,
)


def profile():
    return selected_emulator_matrix().profiles[0]


def task(kind=CurriculumTaskKind.GROUND_UI, evidence=None):
    return CurriculumTask(
        "task",
        "coverage",
        "android_settings_api",
        "android-aosp",
        "abc",
        "root",
        kind,
        "Ground Bluetooth",
        {"display_name": "Bluetooth"},
        evidence or {"snapshot": "abc"},
        setting_id="bluetooth",
    )


def observation(step=0):
    return RolloutObservation(
        "fixture://screen",
        ("Network & internet", "Bluetooth", "Privacy"),
        {"node_count": 12},
        "com.android.settings",
        ".Settings",
        step,
    )


def judgement(kind=ProposalKind.TAP, should_update=True):
    return TeacherJudgement(
        "teacher",
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        should_update,
        TeacherTarget(kind, "Bluetooth", (0.1, 0.1, 0.4, 0.2), "Tap Bluetooth"),
    )


# ---------------------------------------------------------------------------
# project_train pure helpers
# ---------------------------------------------------------------------------


def test_canonical_action_json_round_trip():
    target = judgement().target
    text = canonical_action_json(
        target.preferred_kind, target.target_label, target.bbox_norm, target.response
    )
    assert text.startswith(ACTION_PREFIX)
    completion = teacher_completion_text(target)
    assert completion.startswith(ProposalKind.TAP.value)
    parsed = json.loads(ACTION_PREFIX + completion)
    assert parsed == {
        "kind": "tap",
        "target_label": "Bluetooth",
        "bbox_norm": [0.1, 0.1, 0.4, 0.2],
        "message": "Tap Bluetooth",
    }


def test_parse_action_json_tolerates_trailing_text():
    assert parse_action_json('{"kind": "tap", "a": 1} trailing junk')["kind"] == "tap"
    assert parse_action_json('prefix {"kind": "wait"}')["kind"] == "wait"


def test_parse_action_json_rejects_truncated_and_non_objects():
    assert parse_action_json('{"kind": "ta') is None
    assert parse_action_json("[1, 2, 3]") is None
    assert parse_action_json("no json here") is None


def test_sanitize_bbox_validation():
    assert sanitize_bbox([0.1, 0.1, 0.4, 0.2]) == (0.1, 0.1, 0.4, 0.2)
    assert sanitize_bbox([0.4, 0.1, 0.1, 0.2]) is None  # x0 >= x1
    assert sanitize_bbox([0.1, 0.1, 0.4]) is None
    assert sanitize_bbox([0.1, 0.1, 0.4, 1.5]) is None
    assert sanitize_bbox("not-a-box") is None
    assert sanitize_bbox(None) is None


def test_build_messages_includes_scope_labels_and_settings_context():
    evidence = {
        "os_settings_context": {
            "settings": [
                {
                    "display_name": "Bluetooth",
                    "ui_path": ["Settings", "Connected devices", "Bluetooth"],
                }
            ]
        }
    }
    messages = build_messages(task(evidence=evidence), profile(), observation())
    assert messages[0]["role"] == "system" and "JSON" in messages[0]["content"]
    user = messages[1]["content"]
    assert "Ground Bluetooth" in user
    assert "android-aosp" in user and "Android 10" in user
    assert "Bluetooth, Privacy" in user or "Bluetooth" in user
    assert "- Bluetooth: Settings > Connected devices > Bluetooth" in user
    assert "com.android.settings/.Settings" in user


def test_build_messages_hints_explain_for_informational_tasks():
    messages = build_messages(
        task(kind=CurriculumTaskKind.INFORMATIONAL_QA), profile(), observation()
    )
    assert 'kind "explain"' in messages[1]["content"]


def test_speedrun_lr_warmup_cosine_floor():
    config = {"lr": 4e-4, "warmup_steps": 8, "min_lr_frac": 0.05, "total_steps": 108}
    assert speedrun_lr(0, config) == pytest.approx(4e-4 / 8)
    assert speedrun_lr(7, config) == pytest.approx(4e-4)
    mid = speedrun_lr(58, config)
    assert 0.05 * 4e-4 < mid < 4e-4
    assert speedrun_lr(500, config) == pytest.approx(0.05 * 4e-4)


def _example(prompt, completion, kind=11, student_kind=22, advantage=0.5, box=None):
    return PackedExample(
        prompt_ids=tuple(prompt),
        completion_ids=tuple(completion),
        kind_token=kind,
        student_kind_token=student_kind,
        advantage=advantage,
        box_span=box,
    )


def test_pack_examples_full_blocks_and_kind_anchor():
    examples = [
        _example([1, 2, 3], [11, 4, 5, 0], box=(1, 3)),
        _example([7, 8], [22, 9, 0], kind=22, student_kind=11),
    ]
    blocks = pack_examples(examples, pack_len=16, pad_id=0, rng=random.Random(0))
    assert len(blocks) == 1
    block = blocks[0]
    assert len(block["input_ids"]) == 16 and len(block["labels"]) == 16
    for pos, teacher_kind, student_kind, advantage in block["kind_positions"]:
        # shifted position pos predicts the token at pos + 1
        assert block["input_ids"][pos + 1] == teacher_kind
        assert block["labels"][pos + 1] == teacher_kind
    # prompt positions are masked, completion positions supervised
    first_prompt = block["input_ids"].index(1)
    assert block["labels"][first_prompt] == -100
    # box span stores shifted indexes: labels[i + 1] supervises position i
    (start, end), = block["box_spans"]
    assert [block["labels"][i + 1] for i in range(start, end)] == [4, 5]


def test_pack_examples_flushes_and_pads():
    examples = [_example([1] * 10, [11, 0]), _example([2] * 10, [22, 0])]
    blocks = pack_examples(examples, pack_len=16, pad_id=99, rng=random.Random(0))
    assert len(blocks) == 2
    for block in blocks:
        assert len(block["input_ids"]) == 16
        assert block["input_ids"][-1] == 99  # padded partial block
        assert block["labels"][-1] == -100


def test_pack_examples_truncates_prompt_head_and_skips_oversized():
    examples = [
        _example(list(range(1, 30)), [11, 12, 0]),  # prompt truncated from the head
        _example([5], list(range(20))),  # completion fills the block: skipped
    ]
    blocks = pack_examples(examples, pack_len=16, pad_id=0, rng=random.Random(0))
    assert len(blocks) == 1
    block = blocks[0]
    (kind_pos, teacher_kind, _, _), = block["kind_positions"]
    assert block["input_ids"][kind_pos + 1] == teacher_kind
    assert 29 in block["input_ids"]  # prompt tail kept
    assert 1 not in block["input_ids"]  # prompt head truncated


def test_create_student_resolves_model_id(monkeypatch, tmp_path):
    captured = {}

    class FakeStudent:
        def __init__(self, model_id, **kwargs):
            captured["model_id"] = model_id

    class FakeUpdater:
        def __init__(self, student, checkpoint_dir, **kwargs):
            captured["checkpoint_dir"] = checkpoint_dir

    monkeypatch.setattr(project_train, "QwenSpeedrunStudent", FakeStudent)
    monkeypatch.setattr(project_train, "SpeedrunLoraUpdater", FakeUpdater)
    project_train.create_student({}, tmp_path)
    assert captured["model_id"] == project_train.DEFAULT_MODEL_ID
    project_train.create_student({"model_id": "org/custom-model"}, tmp_path)
    assert captured["model_id"] == "org/custom-model"
    project_train.create_student({"llm": "Qwen/Qwen2.5-1.5B-Instruct"}, tmp_path)
    assert captured["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert captured["checkpoint_dir"] == tmp_path / "checkpoints"


# ---------------------------------------------------------------------------
# project_train torch mechanics (tiny random model, fake tokenizer)
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Deterministic char-level tokenizer; no network, no HF files.

    Char-level tokenization round-trips arbitrary text exactly, which keeps the
    closed action JSON parseable in tests.
    """

    def __init__(self):
        self.vocab = {}
        self.reverse = {}
        self.eos_token_id = self._id("\x00")

    def _id(self, token):
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab) + 1
            self.reverse[self.vocab[token]] = token
        return self.vocab[token]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self._id(char) for char in text]}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        ids = []
        for message in messages:
            ids += self(f"<|im_start|>{message['role']}\n")["input_ids"]
            ids += self(message["content"])["input_ids"]
            ids += self("<|im_end|>\n")["input_ids"]
        if add_generation_prompt:
            ids += self("<|im_start|>assistant\n")["input_ids"]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.reverse[int(i)] for i in ids if int(i) in self.reverse)


@pytest.fixture()
def tiny_student(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    import transformers
    from transformers import Qwen2Config, Qwen2ForCausalLM

    tokenizer = FakeTokenizer()
    config = Qwen2Config(
        vocab_size=4096,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
    )

    def fake_from_pretrained(model_id, **kwargs):
        return Qwen2ForCausalLM(config)

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained
    )
    student = project_train.QwenSpeedrunStudent(
        "tiny-random-qwen2", device="cpu", seed=0
    )
    return student, tokenizer


def _proposal(kind=ProposalKind.TAP, version=0):
    return StudentProposal(
        kind=kind,
        target_label="Bluetooth",
        bbox_norm=None,
        message="Tap Bluetooth",
        confidence=0.7,
        action_distribution=(0.7, 0.1, 0.05, 0.05, 0.05, 0.05),
        log_probability=-0.35,
        policy_version=version,
    )


def _sample(student, kind=ProposalKind.TAP, advantage=0.5, should_update=True):
    return DistillationSample(
        task(),
        profile(),
        observation(),
        _proposal(kind, student.policy_version),
        EnvironmentOutcome(0.0, False, False, None, ""),
        judgement(should_update=should_update),
        0.5,
        advantage,
    )


def test_student_kind_tokens_distinct(tiny_student):
    student, _ = tiny_student
    assert len(set(student.kind_first_tokens.values())) == len(ProposalKind)


def test_propose_fallback_keeps_valid_distribution(tiny_student):
    student, _ = tiny_student
    proposal = student.propose(task(), profile(), observation())
    assert proposal.policy_version == 0
    assert len(proposal.action_distribution) == len(ProposalKind)
    assert sum(proposal.action_distribution) == pytest.approx(1.0)
    assert 0 <= proposal.confidence <= 1


def test_propose_parses_valid_completion(tiny_student, monkeypatch):
    torch = pytest.importorskip("torch")
    student, tokenizer = tiny_student
    completion = 'tap", "target_label": "Bluetooth", "bbox_norm": null, "message": "Tap Bluetooth"}'
    completion_ids = tokenizer(completion)["input_ids"]

    def fake_generate(input_ids=None, **kwargs):
        extra = torch.tensor([completion_ids], dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, extra], dim=1)

    monkeypatch.setattr(student.model, "generate", fake_generate)
    proposal = student.propose(task(), profile(), observation())
    assert proposal.kind is ProposalKind.TAP
    assert proposal.target_label == "Bluetooth"
    assert proposal.message == "Tap Bluetooth"
    assert sum(proposal.action_distribution) == pytest.approx(1.0)


def test_updater_steps_and_writes_adapter_checkpoints(tiny_student, tmp_path):
    torch = pytest.importorskip("torch")
    student, _ = tiny_student
    updater = project_train.SpeedrunLoraUpdater(student, tmp_path, seed=0)
    samples = [_sample(student), _sample(student, kind=ProposalKind.SCROLL)]
    before = [p.detach().clone() for p in student.trainable_parameters()]
    checkpoint = updater.update(samples, OnPolicyConfig())
    assert isinstance(checkpoint, str) and len(checkpoint) == 24
    assert updater.step_count == 1
    assert student.policy_version == 1
    saved = sorted(tmp_path.glob("step-*/adapter_model.safetensors"))
    assert saved and (saved[0].parent / "checkpoint.id").read_text() == checkpoint
    after = list(student.trainable_parameters())
    assert any(not torch.equal(a, b) for a, b in zip(after, before))


def test_updater_rejected_batch_is_a_noop(tiny_student, tmp_path):
    student, _ = tiny_student
    updater = project_train.SpeedrunLoraUpdater(student, tmp_path, seed=0)
    checkpoint = updater.update(
        [_sample(student, should_update=False)], OnPolicyConfig()
    )
    assert len(checkpoint) == 24
    assert updater.step_count == 0
    assert student.policy_version == 0


def test_chunked_ce_matches_materialized_logits(tiny_student):
    torch = pytest.importorskip("torch")
    chunked = project_train._chunked_ce_class()
    rng = torch.Generator().manual_seed(0)
    hidden = torch.randn(17, 32, generator=rng, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(50, 32, generator=rng, dtype=torch.float32)
    labels = torch.randint(0, 50, (17,), generator=rng)
    loss = chunked.apply(hidden, weight, labels, 8)
    reference = torch.nn.functional.cross_entropy(hidden @ weight.T, labels)
    assert loss.item() == pytest.approx(reference.item(), rel=1e-5)
    loss.backward()
    grad_reference = torch.autograd.grad(reference, hidden, retain_graph=False)
    assert torch.allclose(hidden.grad, grad_reference[0], atol=1e-5)


def test_distiller_integration_with_tiny_student(tiny_student, tmp_path):
    pytest.importorskip("torch")
    from android_drlqn.distillation import (
        EmulatorMatrix,
        OnPolicyDistiller,
        ReplayTeacher,
    )

    student, _ = tiny_student
    updater = project_train.SpeedrunLoraUpdater(student, tmp_path, seed=0)
    one_task = task()
    one_profile = profile()

    class OneStepEnvironment:
        def reset(self, task_, profile_):
            return observation()

        def step(self, proposal):
            return EnvironmentOutcome(1.0, True, True, None, "verified")

    key = f"{one_task.task_id}:{one_profile.profile_id}:0"
    teacher = ReplayTeacher({key: judgement()})
    metrics = OnPolicyDistiller(
        student,
        teacher,
        updater,
        lambda _: OneStepEnvironment(),
        OnPolicyConfig(rollout_batch_size=2, max_steps_per_task=1),
    ).train((one_task,), EmulatorMatrix((one_profile,)))
    assert metrics.rollouts == 1
    assert metrics.updates == 1
    assert metrics.checkpoint_id and len(metrics.checkpoint_id) == 24
    assert student.policy_version == 1


# ---------------------------------------------------------------------------
# project_lab
# ---------------------------------------------------------------------------

HIERARCHY_MAIN = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<hierarchy rotation="0">
  <node index="0" text="Network &amp; internet" resource-id="android:id/title"
        class="android.widget.TextView" package="com.android.settings"
        content-desc="" checkable="false" checked="false" clickable="false"
        enabled="true" focusable="false" focused="false" scrollable="false"
        long-clickable="false" password="false" selected="false"
        bounds="[42,240][978,330]"/>
  <node index="1" text="Bluetooth" resource-id="android:id/title"
        class="android.widget.TextView" package="com.android.settings"
        content-desc="" checkable="false" checked="false" clickable="true"
        enabled="true" focusable="true" focused="false" scrollable="false"
        long-clickable="false" password="false" selected="false"
        bounds="[42,340][978,430]"/>
</hierarchy>"""

HIERARCHY_BLUETOOTH = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<hierarchy rotation="0">
  <node index="0" text="Bluetooth" resource-id="android:id/title"
        class="android.widget.TextView" package="com.android.settings"
        content-desc="" checkable="false" checked="false" clickable="false"
        enabled="true" focusable="false" focused="true" scrollable="false"
        long-clickable="false" password="false" selected="false"
        bounds="[42,240][978,330]"/>
  <node index="1" text="Use Bluetooth" resource-id="android:id/widget_text"
        class="android.widget.TextView" package="com.android.settings"
        content-desc="" checkable="true" checked="false" clickable="true"
        enabled="true" focusable="true" focused="false" scrollable="false"
        long-clickable="false" password="false" selected="false"
        bounds="[42,440][978,530]"/>
</hierarchy>"""

PNG_PAYLOAD = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
FOCUS_MAIN = "mCurrentFocus=Window{3f5a3b9 u0 com.android.settings/.Settings}\n"
FOCUS_BT = "mCurrentFocus=Window{aa u0 com.android.settings/.SubSettings}\n"


class FakeAdbRunner:
    """Scripted adb CommandRunner: records calls, serves fixture responses."""

    def __init__(self, qemu="1", hierarchies=(), focuses=()):
        self.calls = []
        self.qemu = qemu
        self.hierarchies = list(hierarchies)
        self.focuses = list(focuses)

    def __call__(self, command, timeout_s):
        self.calls.append(tuple(command))
        if "devices" in command:
            return CommandResult(0, "List of devices attached\nemulator-5554\tdevice\n", "")
        args = tuple(command[3:])
        if args[:2] == ("shell", "getprop"):
            prop = args[2]
            values = {
                "ro.kernel.qemu": self.qemu,
                "ro.build.characteristics": "emulator" if self.qemu == "1" else "nosdcard",
                "ro.build.product": "sdk_gphone64_x86_64" if self.qemu == "1" else "panther",
            }
            return CommandResult(0, values.get(prop, ""), "")
        if args[:2] == ("shell", "wm") and args[2:3] == ("size",):
            return CommandResult(0, "Physical size: 1080x2400", "")
        if args[:2] == ("shell", "uiautomator"):
            return CommandResult(0, "UI hierchary dumped to: /sdcard/window_dump.xml", "")
        if args[:3] == ("exec-out", "cat", "/sdcard/window_dump.xml"):
            text = self.hierarchies[min(1, len(self.hierarchies) - 1)] if self.hierarchies else ""
            if sum(1 for call in self.calls if call[3:5] == ("exec-out", "cat")) <= 1:
                text = self.hierarchies[0] if self.hierarchies else ""
            return CommandResult(0, text, "")
        if args[:2] == ("shell", "dumpsys"):
            count = sum(1 for call in self.calls if "dumpsys" in call)
            if self.focuses:
                return CommandResult(0, self.focuses[min(count - 1, len(self.focuses) - 1)], "")
            return CommandResult(0, FOCUS_MAIN, "")
        return CommandResult(0, "", "")


def fake_binary_runner(adb_path, serial, args, timeout_s):
    assert args[:2] == ("exec-out", "screencap")
    return PNG_PAYLOAD


def make_environment(tmp_path, qemu="1", allow_input=True, hierarchies=None, focuses=None):
    runner = FakeAdbRunner(qemu=qemu, hierarchies=hierarchies or [HIERARCHY_MAIN], focuses=focuses or [FOCUS_MAIN])
    adb = AdbClient(runner=runner)
    env = project_lab.AdbSettingsEnvironment(
        adb=adb,
        serial="emulator-5554",
        screenshots_dir=tmp_path / "shots",
        allow_input=allow_input,
        settle_s=0.0,
        binary_runner=fake_binary_runner,
    )
    return env, runner


def test_parse_ui_hierarchy_extracts_labels_and_bounds():
    nodes = project_lab.parse_ui_hierarchy(HIERARCHY_MAIN)
    assert project_lab.visible_labels(nodes) == ("Network & internet", "Bluetooth")
    bluetooth = nodes[1]
    assert bluetooth.clickable and bluetooth.center == (510, 385)


def test_parse_current_focus():
    assert project_lab.parse_current_focus(FOCUS_MAIN) == (
        "com.android.settings",
        ".Settings",
    )
    assert project_lab.parse_current_focus("mCurrentFocus=null") == ("", "")


def test_label_matches_casefolded_substring():
    assert project_lab.label_matches("bluetooth", ("Bluetooth",))
    assert project_lab.label_matches("Bluetooth", ("Use Bluetooth",))
    assert not project_lab.label_matches("Bluetooth", ("Wi-Fi",))
    assert not project_lab.label_matches("", ("Wi-Fi",))


def test_expected_labels_from_task():
    assert "Bluetooth" in project_lab.expected_labels(task())


def test_environment_requires_emulator_and_input_opt_in(tmp_path):
    env, _ = make_environment(tmp_path, qemu="0")
    with pytest.raises(project_lab.LabEnvironmentError, match="not an emulator"):
        env.reset(task(), profile())
    env, _ = make_environment(tmp_path, allow_input=False)
    with pytest.raises(project_lab.LabEnvironmentError, match="input injection"):
        env.reset(task(), profile())


def test_environment_reset_observation(tmp_path):
    env, runner = make_environment(tmp_path)
    obs = env.reset(task(), profile())
    assert obs.visible_labels == ("Network & internet", "Bluetooth")
    assert obs.package == "com.android.settings" and obs.activity == ".Settings"
    assert obs.step_index == 0
    assert Path(obs.screenshot_reference).read_bytes() == PNG_PAYLOAD
    assert obs.accessibility_summary["node_count"] == 2
    assert any("android.settings.SETTINGS" in " ".join(call) for call in runner.calls)


def test_environment_tap_by_label_verifies_transition(tmp_path):
    env, runner = make_environment(
        tmp_path,
        hierarchies=[HIERARCHY_MAIN, HIERARCHY_BLUETOOTH],
        focuses=[FOCUS_MAIN, FOCUS_BT],
    )
    env.reset(task(), profile())
    outcome = env.step(_proposal(ProposalKind.TAP))
    taps = [call for call in runner.calls if call[3:5] == ("shell", "input") and "tap" in call]
    assert taps, "expected an input tap call"
    assert "510" in taps[0] and "385" in taps[0]  # center of the Bluetooth node
    assert outcome.verified and outcome.terminal and outcome.reward == 1.0


def test_environment_tap_by_bbox_uses_display_size(tmp_path):
    env, runner = make_environment(tmp_path)
    env.reset(task(), profile())
    proposal = StudentProposal(
        ProposalKind.TAP, None, (0.1, 0.1, 0.5, 0.3), "tap", 0.9,
        (0.7, 0.1, 0.05, 0.05, 0.05, 0.05), -0.35, 0,
    )
    env.step(proposal)
    taps = [call for call in runner.calls if "tap" in call]
    assert "324" in taps[0] and "480" in taps[0]  # 0.3*1080, 0.2*2400


def test_environment_unresolvable_tap_penalized(tmp_path):
    env, runner = make_environment(tmp_path)
    env.reset(task(), profile())
    outcome = env.step(
        StudentProposal(
            ProposalKind.TAP, "Nonexistent label", None, "tap", 0.9,
            (0.7, 0.1, 0.05, 0.05, 0.05, 0.05), -0.35, 0,
        )
    )
    assert outcome.reward == -0.05 and not outcome.terminal and not outcome.verified
    assert not any("tap" in call for call in runner.calls)


def test_environment_explain_is_terminal_without_input(tmp_path):
    env, runner = make_environment(tmp_path)
    env.reset(task(kind=CurriculumTaskKind.INFORMATIONAL_QA), profile())
    outcome = env.step(_proposal(ProposalKind.EXPLAIN))
    assert outcome.terminal and not outcome.verified and outcome.next_observation is None
    assert not any("tap" in call or "swipe" in call for call in runner.calls)


def test_environment_scroll_and_back(tmp_path):
    env, runner = make_environment(tmp_path)
    env.reset(task(), profile())
    env.step(_proposal(ProposalKind.SCROLL))
    env.step(_proposal(ProposalKind.BACK))
    commands = [" ".join(call) for call in runner.calls]
    assert any("input swipe" in command for command in commands)
    assert any("input keyevent KEYCODE_BACK" in command for command in commands)


def test_create_environment_serial_resolution(monkeypatch, tmp_path):
    runner = FakeAdbRunner()
    adb = AdbClient(runner=runner)
    env = project_lab.create_environment(
        profile(), serial="emulator-5556", adb=adb, screenshots_dir=tmp_path,
        allow_input=True,
    )
    assert env.serial == "emulator-5556"
    monkeypatch.setenv(project_lab.ENV_SERIAL, "emulator-5558")
    env = project_lab.create_environment(
        profile(), adb=adb, screenshots_dir=tmp_path, allow_input=True
    )
    assert env.serial == "emulator-5558"
    monkeypatch.delenv(project_lab.ENV_SERIAL)
    env = project_lab.create_environment(
        profile(), adb=adb, screenshots_dir=tmp_path, allow_input=True
    )
    assert env.serial == "emulator-5554"  # discovered from adb devices


def test_create_environment_requires_an_emulator(monkeypatch, tmp_path):
    class EmptyRunner(FakeAdbRunner):
        def __call__(self, command, timeout_s):
            if "devices" in command:
                return CommandResult(0, "List of devices attached\n", "")
            return super().__call__(command, timeout_s)

    monkeypatch.delenv(project_lab.ENV_SERIAL, raising=False)
    monkeypatch.delenv(project_lab.ENV_AUTOBOOT, raising=False)
    with pytest.raises(project_lab.LabEnvironmentError, match="no emulator attached"):
        project_lab.create_environment(
            profile(), adb=AdbClient(runner=EmptyRunner()), screenshots_dir=tmp_path
        )
