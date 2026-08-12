from pathlib import Path


def test_mmproj_export_uses_isolated_directory_and_canonicalizes_output():
    script = Path("scripts/export_gui_owl_gguf.sh").read_text(encoding="utf-8")

    assert 'local mmproj="$out_dir/mmproj-${base}-${outtype}.gguf"' in script
    assert 'mmproj_tmp="$(mktemp -d "$out_dir/.mmproj-${base}.XXXXXX")"' in script
    assert 'python "$converter" "$source" --outfile "$mmproj_tmp" --outtype "$outtype" --mmproj' in script
    assert 'mmproj_candidate="$(find "$mmproj_tmp" -maxdepth 1 -type f -name \'*.gguf\' -print -quit)"' in script
    assert 'mv "$mmproj_candidate" "$mmproj"' in script
    assert 'python "$converter" "$source" --outfile "$full" --outtype "$outtype" --mmproj' not in script
    assert 'python "$converter" "$source" --outfile "$mmproj" --outtype "$outtype" --mmproj' not in script


def test_mmproj_export_requires_exactly_one_generated_gguf():
    script = Path("scripts/export_gui_owl_gguf.sh").read_text(encoding="utf-8")

    assert 'mmproj_count="$(find "$mmproj_tmp" -maxdepth 1 -type f -name \'*.gguf\' | wc -l | tr -d \'[:space:]\')"' in script
    assert 'if [[ "$mmproj_count" != "1" ]]; then' in script
    assert 'expected exactly one mmproj GGUF' in script


def test_missing_mmproj_forces_language_rebuild_after_legacy_collision():
    script = Path("scripts/export_gui_owl_gguf.sh").read_text(encoding="utf-8")

    assert 'elif [[ ! -s "$mmproj" ]]; then' in script
    assert 'rebuild_full=1' in script
    assert '"$rebuild_full" == "1" || ! -s "$quant_file"' in script
