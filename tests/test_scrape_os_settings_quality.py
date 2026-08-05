from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "scrape_os_settings.py"
SPEC = importlib.util.spec_from_file_location("scrape_os_settings_quality", MODULE_PATH)
assert SPEC and SPEC.loader
scraper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scraper
SPEC.loader.exec_module(scraper)

def _source_spec(source_key: str = "docs", os_family: str = "android-aosp", confidence_tier: int = 1):
    return scraper.SourceSpec.from_mapping(
        {
            "source_key": source_key,
            "os_family": os_family,
            "root_url": "https://example.com/docs",
            "repository_url": None,
            "confidence_tier": confidence_tier,
            "license_identifier": "TEST",
            "discovery_mode": "single_versioned_page",
            "allowed_hosts": ["example.com"],
            "required_classes": [],
        }
    )


def test_extracts_clean_paths_and_removes_action_language():
    cases = {
        "Open Settings > Privacy > Camera access, then disable the toggle.": (
            "Settings",
            "Privacy",
            "Camera access",
        ),
        "Go to Settings > About phone, tap Build number 7 times.": (
            "Settings",
            "About phone",
            "Build number",
        ),
        "Open Settings > Accounts, tap Add account and pick Murena.": (
            "Settings",
            "Accounts",
            "Add account",
            "Murena",
        ),
        "Open Settings > System > Developer options (enabled previously).": (
            "Settings",
            "System",
            "Developer options",
        ),
        "Open Settings > Security > App verification and enable it.": (
            "Settings",
            "Security",
            "App verification",
        ),
        "Open Settings > Apps > Open by default.": (
            "Settings",
            "Apps",
            "Open by default",
        ),
        "Open Settings > Apps > Special app access > Allow from this source.": (
            "Settings",
            "Apps",
            "Special app access",
            "Allow from this source",
        ),
    }
    for text, expected in cases.items():
        assert [candidate.path for candidate in scraper.extract_ui_paths(text)] == [expected]


def test_rejects_technical_non_android_and_sentence_segments():
    rejected = (
        'Open Settings > controller="com.android.settings.foo" > PreferenceScreen.',
        "Open Settings > your name > iCloud.",
        "Open Settings > system > alarm_alert will be used.",
        "Open Settings > wait a moment before trying again.",
        "Open Settings > <TextView android:id='title'>.",
        "Configure Privacy settings > Camera access.",
        "Open Settings > Privacy > Camera access or Settings > Location > App access.",
    )
    for text in rejected:
        assert scraper.extract_ui_paths(text) == ()


def test_code_api_and_resource_documents_are_evidence_only():
    source = _source_spec()
    documents = (
        scraper.RawDocument(
            "android_settings_api",
            "android-aosp",
            "https://developer.android.com/reference/android/provider/Settings",
            b"<html><p>Open Settings &gt; Privacy &gt; Camera access.</p></html>",
            "text/html",
        ),
        scraper.RawDocument(
            "aosp_settings_source",
            "android-aosp",
            "git+https://example.com/repo@abc#res/xml/privacy.xml",
            b"<PreferenceScreen>Open Settings > Privacy > Camera access</PreferenceScreen>",
            "application/xml",
            relative_path="res/xml/privacy.xml",
        ),
        scraper.RawDocument(
            "aosp_settings_source",
            "android-aosp",
            "git+https://example.com/repo@abc#src/Settings.kt",
            b'val help = "Open Settings > Privacy > Camera access"',
            "text/plain",
            relative_path="src/Settings.kt",
        ),
    )
    assert all(scraper.normalize_document(source, document)["claims"] == [] for document in documents)


def test_markdown_fences_and_html_pre_blocks_do_not_emit_claims():
    source = _source_spec()
    markdown = scraper.RawDocument(
        "docs",
        "android-aosp",
        "git+https://example.com/repo@abc#guide.md",
        b"# Guide\n```text\nOpen Settings > Privacy > Camera access\n```\n",
        "text/markdown",
        relative_path="guide.md",
    )
    html = scraper.RawDocument(
        "docs",
        "android-aosp",
        "https://example.com/docs",
        b"<html><h1>Guide</h1><pre>Open Settings > Privacy > Camera access</pre></html>",
        "text/html",
    )
    assert scraper.normalize_document(source, markdown)["claims"] == []
    assert scraper.normalize_document(source, html)["claims"] == []


def test_display_name_and_hint_come_from_validated_terminal_path():
    source = _source_spec()
    document = scraper.RawDocument(
        "docs",
        "android-aosp",
        "https://example.com/docs",
        (
            b"<html><h1>How to Enable the OTA Test Channel Server</h1>"
            b"<p>Open Settings &gt; About phone, tap Build number 7 times.</p></html>"
        ),
        "text/html",
    )
    normalized = scraper.normalize_document(source, document)
    assert len(normalized["claims"]) == 1
    claim = normalized["claims"][0]
    assert claim["display_name"] == "Build number"
    assert claim["ui_path"] == ["Settings", "About phone", "Build number"]
    assert claim["canonical_setting_hint"] == "android-aosp.about_phone_build_number"


def _normalized_document(source_id, confidence_tier, path, display_name=None):
    section_id = f"section-{source_id}"
    claim = {
        "claim_kind": "setting_variant",
        "canonical_setting_hint": "android-aosp.privacy_camera_access",
        "display_name": display_name or path[-1],
        "ui_path": list(path),
        "control_plane": "settings_ui",
        "required_role": "user",
        "risk_level": "medium",
        "scope": {
            "os_family": "android-aosp",
            "source_key": source_id,
            "canonical_url": f"https://example.com/{source_id}",
            "section_id": section_id,
        },
    }
    return {
        "document": {
            "source_id": source_id,
            "document_id": f"doc-{source_id}",
            "canonical_url": f"https://example.com/{source_id}",
            "confidence_tier": confidence_tier,
            "os_family": "android-aosp",
        },
        "claims": [claim],
        "sections": [{"section_id": section_id, "claim_indexes": [0]}],
    }


def test_quality_gate_removes_invalid_claims_deduplicates_and_remaps_indexes():
    lower_quality = _normalized_document(
        "tier1", 1, ("Settings", "Privacy", "Camera access")
    )
    winner = _normalized_document(
        "tier0", 0, ("Settings", "Privacy", "Camera access")
    )
    invalid = _normalized_document(
        "invalid", 0, ("Settings", 'controller="com.android.settings"')
    )

    output, report = scraper.sanitize_and_deduplicate_documents(
        [lower_quality, winner, invalid]
    )

    assert report.input_claim_count == 3
    assert report.emitted_claim_count == 1
    assert report.invalid_claim_count == 1
    assert report.duplicate_claim_count == 1
    assert output[0]["claims"] == []
    assert output[0]["sections"][0]["claim_indexes"] == []
    assert output[1]["claims"][0]["display_name"] == "Camera access"
    assert output[1]["sections"][0]["claim_indexes"] == [0]
    assert output[2]["claims"] == []
    assert output[2]["sections"][0]["claim_indexes"] == []


def test_existing_legacy_shapes_fail_the_audit_gate():
    heading_as_name = _normalized_document(
        "legacy", 1, ("Settings", "About phone", "Build number"),
        display_name="How to Enable the OTA Test Channel Server",
    )
    report = scraper.audit_normalized_documents([heading_as_name])
    assert report.invalid_claim_count == 1
    assert dict(report.invalid_reasons)["display_name_mismatch"] == 1


def test_quality_gate_reports_broken_section_indexes():
    document = _normalized_document(
        "broken", 1, ("Settings", "Privacy", "Camera access")
    )
    document["sections"][0]["claim_indexes"] = [0, 99, "zero"]
    output, report = scraper.sanitize_and_deduplicate_documents([document])
    assert report.structural_error_count == 2
    assert dict(report.invalid_reasons)["broken_section_index"] == 2
    assert output[0]["sections"][0]["claim_indexes"] == [0]


def test_quality_gate_rejects_scope_mismatch_and_unreferenced_claim():
    document = _normalized_document(
        "scope", 1, ("Settings", "Privacy", "Camera access")
    )
    document["claims"][0]["scope"]["os_family"] = "grapheneos"
    document["sections"][0]["claim_indexes"] = []
    output, report = scraper.sanitize_and_deduplicate_documents([document])
    reasons = dict(report.invalid_reasons)
    assert report.invalid_claim_count == 1
    assert reasons["scope_os_mismatch"] == 1
    assert reasons["unreferenced_claim"] == 1
    assert output[0]["claims"] == []


def test_extracts_ordered_calyx_style_settings_procedure():
    blocks = (
        scraper.TextBlock(
            ("Bluetooth timeout",),
            (
                "1. Open Settings from the application page "
                "2. Select Connected Devices "
                "3. Select Bluetooth timeout "
                "4. Select the timeout timeframe"
            ),
        ),
    )
    candidates = scraper.extract_stepwise_ui_paths(blocks)
    assert [candidate.path for candidate in candidates] == [
        ("Settings", "Connected Devices", "Bluetooth timeout")
    ]


def test_html_ordered_steps_emit_claim_without_breadcrumb_symbols():
    source = _source_spec(source_key="calyxos_official_docs", os_family="calyxos")
    document = scraper.RawDocument(
        "calyxos_official_docs",
        "calyxos",
        "https://calyxos.org/docs/guide/getting-started/",
        (
            b"<html><h2>Bluetooth timeout</h2><ol>"
            b"<li>Open Settings from the application page</li>"
            b"<li>Select Connected Devices</li>"
            b"<li>Select Bluetooth timeout</li>"
            b"<li>Select the timeout timeframe</li>"
            b"</ol></html>"
        ),
        "text/html",
    )
    normalized = scraper.normalize_document(source, document)
    assert len(normalized["claims"]) == 1
    assert normalized["claims"][0]["ui_path"] == [
        "Settings",
        "Connected Devices",
        "Bluetooth timeout",
    ]


def test_ordered_procedure_requires_explicit_settings_start():
    blocks = (
        scraper.TextBlock(
            (),
            "1. Select Connected Devices 2. Select Bluetooth timeout",
        ),
    )
    assert scraper.extract_stepwise_ui_paths(blocks) == ()


def test_require_all_os_checks_evidence_not_synthetic_claims():
    source = _source_spec(source_key="api", os_family="android-aosp")
    documents = [
        {
            "document": {
                "source_id": "api",
                "document_id": "doc-api",
                "canonical_url": "https://example.com/api",
                "confidence_tier": 1,
                "os_family": "android-aosp",
            },
            "claims": [],
            "sections": [],
        }
    ]
    scraper.validate_corpus_coverage(
        [source], documents, require_all_os=True, required_claim_os=[]
    )


def test_require_claim_os_fails_only_when_explicitly_requested():
    import pytest

    source = _source_spec(source_key="api", os_family="android-aosp")
    documents = [
        {
            "document": {
                "source_id": "api",
                "document_id": "doc-api",
                "canonical_url": "https://example.com/api",
                "confidence_tier": 1,
                "os_family": "android-aosp",
            },
            "claims": [],
            "sections": [],
        }
    ]
    with pytest.raises(scraper.ScrapeError, match="no validated setting_variant claims"):
        scraper.validate_corpus_coverage(
            [source],
            documents,
            require_all_os=True,
            required_claim_os=["android-aosp"],
        )


def test_rejects_action_instruction_placeholder_and_ui_chrome_segments():
    rejected = (
        "toggle Connect to /e/OS test channel on",
        "refresh using top-right arrow if needed",
        "Generate New Code , then Set",
        "click the related Wi-Fi name",
        "see all xxx apps",
        "3 dots on top right",
        "Accounts again and confirm that your Google account is listed",
        "to continue",
        "the tab related to the SIM card you are troubleshooting",
        "your Murena account",
        "your Murena Workspace email address",
        "profile",
    )
    for value in rejected:
        assert scraper.normalize_ui_segment(value) is None


def test_action_and_placeholder_breadcrumbs_do_not_emit_claims():
    rejected = (
        "Open Settings > System > Developer options > toggle Connect to /e/OS test channel on.",
        "Open Settings > Updater > refresh using top-right arrow if needed.",
        "Open Settings > Passwords & accounts > your Murena account > Account sync.",
        "Open Settings > 3 dots > Copy link.",
        "Open Settings > System > Advanced > microG > Google device registration > profile.",
    )
    for text in rejected:
        assert scraper.extract_ui_paths(text) == ()


def test_document_deduplication_prefers_the_copy_with_valid_claims():
    empty = _normalized_document(
        "eos_official_docs", 1, ("Settings", "Updater")
    )
    empty["claims"] = []
    empty["sections"][0]["claim_indexes"] = []
    empty["document"]["canonical_url"] = "https://doc.e.foundation/os/how-to/emulator"

    useful = _normalized_document(
        "eos_emulator_docs", 1, ("Settings", "System", "Developer options")
    )
    useful["document"]["canonical_url"] = "https://doc.e.foundation/os/how-to/emulator"
    useful["claims"][0]["scope"]["canonical_url"] = useful["document"]["canonical_url"]

    output, report = scraper.sanitize_and_deduplicate_documents([empty, useful])

    assert report.duplicate_document_count == 1
    assert len(output) == 1
    assert output[0]["document"]["source_id"] == "eos_emulator_docs"
    assert output[0]["claims"][0]["display_name"] == "Developer options"


def test_global_section_ids_must_be_unique_across_documents():
    first = _normalized_document(
        "first", 1, ("Settings", "Privacy", "Camera access")
    )
    second = _normalized_document(
        "second", 1, ("Settings", "Location", "App access")
    )
    second["sections"][0]["section_id"] = first["sections"][0]["section_id"]
    second["claims"][0]["scope"]["section_id"] = first["sections"][0]["section_id"]

    _, report = scraper.sanitize_and_deduplicate_documents([first, second])

    reasons = dict(report.invalid_reasons)
    assert report.structural_error_count == 1
    assert reasons["duplicate_global_section_id"] == 1


def test_clear_storage_is_not_classified_as_low_risk():
    assert scraper.infer_risk_level("Open Storage & cache and tap Clear storage") == "medium"


def test_quality_gate_rejects_destructive_claim_with_low_risk():
    document = _normalized_document(
        "destructive", 1, ("Settings", "Apps", "Storage & cache", "Clear storage")
    )
    document["claims"][0]["risk_level"] = "low"
    output, report = scraper.sanitize_and_deduplicate_documents([document])
    assert output[0]["claims"] == []
    assert dict(report.invalid_reasons)["risk_level_too_low"] == 1


def test_document_deduplication_does_not_prefer_broken_evidence():
    broken = _normalized_document(
        "broken-copy", 0, ("Settings", "System", "Developer options")
    )
    broken["document"]["canonical_url"] = "https://example.com/same-page"
    broken["sections"][0]["claim_indexes"] = [99]

    clean = _normalized_document(
        "clean-copy", 1, ("Settings", "System", "Developer options")
    )
    clean["document"]["canonical_url"] = "https://example.com/same-page"

    output, report = scraper.sanitize_and_deduplicate_documents([broken, clean])

    assert report.duplicate_document_count == 1
    assert len(output) == 1
    assert output[0]["document"]["source_id"] == "clean-copy"

