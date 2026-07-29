from __future__ import annotations

import importlib.util
import json
import threading
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "scrape_os_settings.py"
SPEC = importlib.util.spec_from_file_location("scrape_os_settings", MODULE_PATH)
assert SPEC and SPEC.loader
scraper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scraper
SPEC.loader.exec_module(scraper)


class FixtureHandler(BaseHTTPRequestHandler):
    pages: dict[str, bytes] = {}

    def do_GET(self):  # noqa: N802
        if self.path == "/robots.txt":
            body = b"User-agent: *\nDisallow:\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = self.pages.get(self.path)
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A003
        return


@pytest.fixture
def server():
    FixtureHandler.pages = {}
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def _registry(tmp_path: Path, server: ThreadingHTTPServer) -> Path:
    port = server.server_address[1]
    sources = []
    for family in ("android-aosp", "grapheneos", "calyxos", "eos"):
        key = family.replace("-", "_")
        path = f"/{key}/index.html"
        FixtureHandler.pages[path] = (
            f"<html><h1>{family} Settings</h1>"
            f"<p>Open Settings &gt; Privacy &gt; Camera access, then disable the toggle.</p></html>"
        ).encode()
        sources.append(
            {
                "source_key": key,
                "os_family": family,
                "root_url": f"http://127.0.0.1:{port}{path}",
                "repository_url": None,
                "confidence_tier": 1,
                "license_identifier": "TEST",
                "discovery_mode": "single_versioned_page",
                "allowed_hosts": ["127.0.0.1"],
                "required_classes": ["settings_ui_path"],
            }
        )
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"sources": sources}), encoding="utf-8")
    return path


def _args(tmp_path: Path, sources: Path):
    return scraper.build_parser().parse_args(
        [
            "--sources",
            str(sources),
            "--all-enabled-os",
            "--snapshot-dir",
            str(tmp_path / "snapshots"),
            "--output",
            str(tmp_path / "normalized.json"),
            "--require-all-os",
            "--fail-on-stale-source",
            "--delay-seconds",
            "0",
        ]
    )


def test_scrapes_all_os_into_one_json_array(tmp_path, server):
    sources = _registry(tmp_path, server)
    args = _args(tmp_path, sources)
    manifest = scraper.run(args)

    documents = json.loads((tmp_path / "normalized.json").read_text(encoding="utf-8"))
    assert isinstance(documents, list)
    assert {item["document"]["os_family"] for item in documents} == {
        "android-aosp",
        "grapheneos",
        "calyxos",
        "eos",
    }
    assert all(item["claims"] for item in documents)
    assert all(item["claims"][0]["claim_kind"] == "setting_variant" for item in documents)
    assert all(item["claims"][0]["ui_path"][:2] == ["Settings", "Privacy"] for item in documents)
    assert manifest["document_count"] == 4
    assert manifest["setting_claim_count"] == 4
    assert len(list((tmp_path / "snapshots" / "raw").rglob("*.*"))) == 4


def test_stale_source_gate_preserves_previous_manifest(tmp_path, server):
    sources = _registry(tmp_path, server)
    args = _args(tmp_path, sources)
    scraper.run(args)
    manifest_path = tmp_path / "snapshots" / "manifest.json"
    original_manifest = manifest_path.read_text(encoding="utf-8")

    FixtureHandler.pages["/android_aosp/index.html"] = (
        b"<html><h1>Changed Settings</h1><p>Open Settings > Security > App verification and enable it.</p></html>"
    )
    with pytest.raises(scraper.StaleSourceError):
        scraper.run(args)
    assert manifest_path.read_text(encoding="utf-8") == original_manifest


def test_rejects_non_allowlisted_source():
    with pytest.raises(scraper.AllowlistError):
        scraper.SourceSpec.from_mapping(
            {
                "source_key": "bad",
                "os_family": "bad-os",
                "root_url": "https://example.com/docs",
                "confidence_tier": 1,
                "license_identifier": "TEST",
                "discovery_mode": "official_index",
                "allowed_hosts": ["docs.example.org"],
                "required_classes": [],
            }
        )


def test_missing_required_class_fails(tmp_path, server):
    sources = _registry(tmp_path, server)
    payload = json.loads(sources.read_text(encoding="utf-8"))
    payload["sources"][0]["required_classes"] = ["release_notes"]
    sources.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(scraper.ScrapeError, match="required document classes"):
        scraper.run(_args(tmp_path, sources))
