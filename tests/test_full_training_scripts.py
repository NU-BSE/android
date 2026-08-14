from pathlib import Path


SCRIPTS = {
    "1.5b": Path("scripts/train_gui_owl_1_5b_full.sh"),
    "1b": Path("scripts/train_gui_owl_1b_full.sh"),
    "0.5b": Path("scripts/train_gui_owl_0_5b_full.sh"),
}


def test_full_training_scripts_never_convert_to_gguf():
    for size, path in SCRIPTS.items():
        text = path.read_text(encoding="utf-8")
        assert "run_gui_owl_progressive_distillation.py" in text, size
        assert f"package_gui_owl_hf_weights.sh {size}" in text, size
        assert "export_gui_owl_gguf.sh" not in text, size
        assert "llama.cpp" not in text, size
        assert "llama-quantize" not in text, size


def test_full_training_scripts_require_generation_acceptance():
    for size, path in SCRIPTS.items():
        text = path.read_text(encoding="utf-8")
        assert 'value.get("accepted") is True' in text, size
        assert f"artifacts/students/v2/gui-owl-{size}-distilled" in text, size


def test_smaller_students_require_accepted_1_5b_predecessor():
    predecessor = "artifacts/students/v2/gui-owl-1.5b-distilled"
    for size in ("1b", "0.5b"):
        text = SCRIPTS[size].read_text(encoding="utf-8")
        assert predecessor in text
        assert 'if ! accepted "$predecessor"; then' in text


def test_hf_packager_contains_only_final_checkpoint_contract():
    text = Path("scripts/package_gui_owl_hf_weights.sh").read_text(encoding="utf-8")
    assert 'artifacts/students/v2/gui-owl-${student}-distilled' in text
    assert "artifacts/gguf" not in text
    assert "export_gui_owl_gguf" not in text
    assert "llama.cpp" in text  # only in the explicit statement that it is not called
    assert 'value.get("accepted") is not True' in text
