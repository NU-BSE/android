from pathlib import Path


def _script() -> str:
    return Path("scripts/export_gui_owl_gguf.sh").read_text(encoding="utf-8")


def test_export_resolves_v2_source_from_student_config_and_requires_acceptance():
    script = _script()
    assert 'student_config="${STUDENT_CONFIG:-$repo_root/config/gui_owl_students.json}"' in script
    assert 'source_rel="$(resolve_source "$size")"' in script
    assert 'check_acceptance "$source"' in script
    assert 'value.get("accepted") is not True' in script
    assert 'GGUF_ALLOW_UNACCEPTED' in script
    assert 'artifacts/gguf/v2' in script


def test_mmproj_export_uses_isolated_directory_and_canonicalizes_output():
    script = _script()
    assert 'mmproj="$out_dir/mmproj-${base}-${outtype}.gguf"' in script
    assert 'mmproj_tmp="$(mktemp -d "$out_dir/.mmproj-${base}.XXXXXX")"' in script
    assert 'python "$converter" "$source" --outfile "$mmproj_tmp" --outtype "$outtype" --mmproj' in script
    assert 'mmproj_candidate="$(find "$mmproj_tmp" -maxdepth 1 -type f -name \'*.gguf\' -print -quit)"' in script
    assert 'mv "$mmproj_candidate" "$mmproj"' in script
    assert 'python "$converter" "$source" --outfile "$full" --outtype "$outtype" --mmproj' not in script
    assert 'python "$converter" "$source" --outfile "$mmproj" --outtype "$outtype" --mmproj' not in script


def test_mmproj_export_requires_exactly_one_generated_gguf():
    script = _script()
    assert 'mmproj_count="$(find "$mmproj_tmp" -maxdepth 1 -type f -name \'*.gguf\' | wc -l | tr -d \'[:space:]\')"' in script
    assert 'if [[ "$mmproj_count" != "1" ]]; then' in script
    assert 'expected exactly one mmproj GGUF' in script


def test_missing_mmproj_forces_language_rebuild_after_legacy_collision():
    script = _script()
    assert 'elif [[ ! -s "$mmproj" ]]; then' in script
    assert 'rebuild_full=1' in script
    assert '"$rebuild_full" == "1" || ! -s "$quant_file"' in script


def test_packager_uses_configured_distilled_checkpoint_and_v2_dirs():
    script = Path("scripts/package_gui_owl_students.sh").read_text(encoding="utf-8")
    assert 'hf_rel="$(resolve_source_rel "$size")"' in script
    assert 'artifacts/transfer/v2' in script
    assert 'artifacts/gguf/v2' in script
