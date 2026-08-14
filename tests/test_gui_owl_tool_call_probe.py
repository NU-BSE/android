import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ablate_gui_owl_text_layers as ablate  # noqa: E402
import probe_gui_owl_tool_call_rank as probe  # noqa: E402


def test_image_paths_are_repointed_only_for_image_parts():
    messages = [
        {"role": "system", "content": "operate android"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/workspace/android/artifacts/shot.png"},
                {"type": "image", "image": "/elsewhere/shot.png"},
                {"type": "text", "text": "/workspace/android/ is not a path here"},
            ],
        },
    ]
    rewritten = probe._rewrite_images(
        messages, source_prefix="/workspace/android/", target_prefix="/local/root/"
    )
    parts = rewritten[1]["content"]
    assert parts[0]["image"] == "/local/root/artifacts/shot.png"
    assert parts[1]["image"] == "/elsewhere/shot.png"
    assert parts[2]["text"] == "/workspace/android/ is not a path here"
    # The caller's records must not be mutated in place.
    assert messages[1]["content"][0]["image"] == "/workspace/android/artifacts/shot.png"


def test_record_selection_is_deterministic_for_a_seed():
    records = [{"messages": [{"role": "user"}, {"role": "assistant"}], "id": i} for i in range(200)]
    first = probe._select_records(records, seed=0, holdout_fraction=0.05, max_records=4)
    second = probe._select_records(records, seed=0, holdout_fraction=0.05, max_records=4)
    other = probe._select_records(records, seed=1, holdout_fraction=0.05, max_records=4)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert [row["id"] for row in first] != [row["id"] for row in other]


def test_variants_accept_named_and_bare_layer_lists():
    assert ablate._parse_variants(["depth16=11,15,22,24"]) == [
        ("depth16", (11, 15, 22, 24))
    ]
    assert ablate._parse_variants(["22,24"]) == [("drop-22,24", (22, 24))]
    with pytest.raises(ValueError, match="drops no layer"):
        ablate._parse_variants(["empty="])


def test_original_layer_ids_come_from_the_compression_lineage(tmp_path):
    checkpoint = tmp_path / "depth-20"
    checkpoint.mkdir()
    ids = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 24, 27]
    (checkpoint / "compression-lineage.json").write_text(
        json.dumps({"text_original_layer_ids": ids}), encoding="utf-8"
    )
    assert ablate._original_layer_ids(checkpoint, 20) == ids
    # A lineage that disagrees with the loaded stack would silently mislabel
    # every ablation, so it must fail instead of being reinterpreted.
    with pytest.raises(ValueError, match="16"):
        ablate._original_layer_ids(checkpoint, 16)


def test_missing_lineage_falls_back_to_stack_positions(tmp_path):
    checkpoint = tmp_path / "bare"
    checkpoint.mkdir()
    assert ablate._original_layer_ids(checkpoint, 4) == [0, 1, 2, 3]
