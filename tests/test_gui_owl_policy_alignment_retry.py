from __future__ import annotations

import json
from pathlib import Path


def _plan() -> dict[str, object]:
    return json.loads(Path("config/gui_owl_students.json").read_text(encoding="utf-8"))


def test_on_policy_must_learn_gui_syntax_before_specialization():
    plan = _plan()
    on_policy = plan["stages"]["on_policy"]
    gate = on_policy["generation_gate"]
    assert float(gate["gui_parse_rate_min"]) >= 0.5
    assert float(on_policy["supervised_weight"]) >= 0.3
    assert float(on_policy["forward_kl_weight"]) >= 0.25
    assert float(plan["specialized_corpora"]["gui_policy"]["weight"]) >= 2.0
    assert int(on_policy["eval_interval_updates"]) <= 100


def test_k3_sampling_does_not_hide_first_token_eos():
    """The student must be allowed to sample EOS so reverse-KL can penalize it.

    The failed policy bridge used min_new_tokens=4. Greedy evaluation could emit
    EOS immediately, but the student-sampled K3 path forcibly suppressed that
    action and therefore never trained on the failure the gate was measuring.
    """

    stages = _plan()["stages"]
    assert int(stages["on_policy"]["on_policy_min_new_tokens"]) == 0
    assert int(stages["specialize"]["on_policy_min_new_tokens"]) == 0


def test_specialization_keeps_an_on_policy_anchor():
    stage = _plan()["stages"]["specialize"]
    assert float(stage["k3_weight"]) > 0.0
    assert float(stage["k3_delta_clip"]) <= 6.0
    assert int(stage["on_policy_min_new_tokens"]) == 0
    assert float(stage["generation_gate"]["gui_parse_rate_min"]) >= 0.5


def test_targeted_retry_reuses_depth_16_and_never_exports_gguf():
    text = Path("scripts/retry_gui_owl_1_5b_policy_alignment.sh").read_text(encoding="utf-8")
    assert "depth-16-recovered" in text
    assert "gui-bridge" in text
    assert "--stage on_policy" in text
    assert text.count("--stage specialize") >= 2
    assert "--max-records-per-corpus" in text
    assert "distill_gui_owl_policy_aligned.py" in text
    assert "run_gui_owl_progressive_distillation.py" not in text
    assert "export_gui_owl_gguf.sh" not in text
    assert "llama-quantize" not in text


def test_retry_preflights_the_actual_gui_teacher_before_archiving_or_training():
    text = Path("scripts/retry_gui_owl_1_5b_policy_alignment.sh").read_text(encoding="utf-8")
    teacher = text.index("evaluate_gui_owl_policy_teacher.py")
    archive = text.index("archived previous policy checkpoints")
    bridge = text.index("GUI syntax bridge from accepted depth-16 recovery")
    assert teacher < archive < bridge
    assert "teacher-gui.jsonl.summary.json" in text


def test_policy_aligned_entrypoint_stratifies_gui_holdout():
    text = Path("scripts/distill_gui_owl_policy_aligned.py").read_text(encoding="utf-8")
    assert "MIN_GUI_EVAL_RECORDS = 8" in text
    assert "record.mode == \"gui_policy\"" in text
    assert "base._evaluate_generation = _stratified_evaluate_generation" in text


def test_policy_diagnostic_saves_concrete_generations():
    text = Path("scripts/evaluate_gui_owl_policy_checkpoint.py").read_text(encoding="utf-8")
    assert '"generation"' in text
    assert '"gui_parse_valid"' in text
    assert '"target"' in text
    assert "parse_gui_owl_output" in text


def test_teacher_policy_preflight_uses_named_peft_adapter_and_saves_generations():
    text = Path("scripts/evaluate_gui_owl_policy_teacher.py").read_text(encoding="utf-8")
    assert "resolve_peft_adapter_dir" in text
    assert "PeftModel.from_pretrained" in text
    assert "model.set_adapter(args.adapter_name)" in text
    assert '"generation"' in text
    assert '"gui_parse_valid"' in text
