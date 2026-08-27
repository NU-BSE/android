import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import distill_gui_owl_student as trainer  # noqa: E402


def _records(broad: int, gui: int) -> list[trainer.TrainingRecord]:
    # Mirrors the real mix: the broad recovery corpus dwarfs the GUI corpus, so
    # the head of a shuffled holdout is almost all broad records.
    values = [
        trainer.TrainingRecord("broad", "base", [{"role": "user"}], 1.0)
        for _ in range(broad)
    ]
    values.extend(
        trainer.TrainingRecord("gui_policy", "gui_policy", [{"role": "user"}], 1.0)
        for _ in range(gui)
    )
    return values


def test_evaluation_reserves_gui_records_a_plain_head_would_miss():
    records = _records(broad=300, gui=15)
    assert all(record.mode != "gui_policy" for record in records[:32])

    selected = trainer._select_evaluation_records(records, 32)
    assert len(selected) == 32
    gui = [record for record in selected if record.mode == "gui_policy"]
    assert len(gui) >= trainer.MIN_GUI_EVAL_RECORDS


def test_evaluation_without_any_gui_record_is_unchanged():
    # Stage A holds out broad data only; it must not invent a GUI sample.
    records = _records(broad=100, gui=0)
    selected = trainer._select_evaluation_records(records, 32)
    assert len(selected) == 32
    assert all(record.mode != "gui_policy" for record in selected)


def test_evaluation_budget_larger_than_the_holdout_keeps_every_record():
    records = _records(broad=10, gui=3)
    assert trainer._select_evaluation_records(records, 32) == records


def test_gui_quota_never_starves_the_budget_when_gui_dominates():
    records = _records(broad=2, gui=50)
    selected = trainer._select_evaluation_records(records, 8)
    assert len(selected) == 8
