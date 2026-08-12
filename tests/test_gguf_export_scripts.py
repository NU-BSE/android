from pathlib import Path


def test_mmproj_export_uses_dedicated_output_path():
    script = Path("scripts/export_gui_owl_gguf.sh").read_text(encoding="utf-8")

    assert 'local mmproj="$out_dir/mmproj-${base}-${outtype}.gguf"' in script
    assert 'python "$converter" "$source" --outfile "$mmproj" --outtype "$outtype" --mmproj' in script
    assert 'python "$converter" "$source" --outfile "$full" --outtype "$outtype" --mmproj' not in script


def test_missing_mmproj_forces_language_rebuild_after_legacy_collision():
    script = Path("scripts/export_gui_owl_gguf.sh").read_text(encoding="utf-8")

    assert 'elif [[ ! -s "$mmproj" ]]; then' in script
    assert 'rebuild_full=1' in script
    assert '"$rebuild_full" == "1" || ! -s "$quant_file"' in script
