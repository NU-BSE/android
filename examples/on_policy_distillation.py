"""Minimal replay-backed on-policy distillation example."""
from android_drlqn import (
    CurriculumTask, CurriculumTaskKind, EnvironmentOutcome, EmulatorMatrix,
    NumpyLinearStudent, NumpyPolicyUpdater, OnPolicyConfig, OnPolicyDistiller,
    ProposalKind, ReplayTeacher, RolloutObservation, TeacherJudgement,
    TeacherTarget, selected_emulator_matrix,
)


class Environment:
    def reset(self, task, profile):
        return RolloutObservation("fixture://settings", ("Bluetooth",), {}, "com.android.settings", ".Settings", 0)

    def step(self, proposal):
        return EnvironmentOutcome(1.0, True, True, None, "fixture verified")


task = CurriculumTask(
    task_id="bluetooth-grounding",
    coverage_id="fixture-coverage",
    source_key="android_settings_api",
    os_family="android-aosp",
    snapshot_sha256="fixture",
    section_id="bluetooth",
    kind=CurriculumTaskKind.GROUND_UI,
    prompt="Ground Bluetooth",
    expected_output={"display_name": "Bluetooth"},
    evidence={"snapshot_sha256": "fixture"},
)
profile = selected_emulator_matrix().profiles[0]
student = NumpyLinearStudent(seed=0)
teacher = ReplayTeacher({
    f"{task.task_id}:{profile.profile_id}:0": TeacherJudgement(
        "replay", 1, 1, 1, 1, 1, 1, True,
        TeacherTarget(ProposalKind.TAP, "Bluetooth", (0.1, 0.1, 0.5, 0.2), "Tap Bluetooth"),
    )
})
metrics = OnPolicyDistiller(
    student,
    teacher,
    NumpyPolicyUpdater(student),
    lambda _: Environment(),
    OnPolicyConfig(rollout_batch_size=1, max_steps_per_task=1),
).train((task,), EmulatorMatrix((profile,)))
print(metrics)
