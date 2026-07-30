#!/usr/bin/env python3
"""Snapshot and normalize OS settings documentation into ORM JSON.

The crawler is deliberately bounded and fail-closed:

* only source-registry hosts are reachable;
* robots.txt is respected for HTTP sources;
* redirects are revalidated against the same allowlist;
* request count, depth, response size and delay are bounded;
* raw source bytes are stored under a content-addressed snapshot directory;
* a changed source revision can be rejected with ``--fail-on-stale-source``;
* output is one deterministic JSON array suitable for DocumentationCurriculumBuilder.

The script does not execute documentation, download Android images or invoke a
model. Normalization is rule based and emits evidence-linked ``setting_variant``
claims only when a section contains a settings path or an actionable settings
instruction.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse
from urllib.request import Request, build_opener
from urllib.robotparser import RobotFileParser

SCHEMA_VERSION = 1
DEFAULT_USER_AGENT = "NU-BSE-OS-Settings-Snapshotter/1.0 (+documentation research)"
TEXT_CONTENT_TYPES = (
    "text/html",
    "text/plain",
    "text/markdown",
    "text/xml",
    "application/xml",
    "application/xhtml+xml",
    "application/json",
)
TEXT_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".gradle",
    ".h",
    ".html",
    ".htm",
    ".java",
    ".json",
    ".kt",
    ".kts",
    ".md",
    ".properties",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
RELEVANT_PATH_PARTS = (
    "setting",
    "permission",
    "privacy",
    "security",
    "guide",
    "docs",
    "help",
    "release",
    "device",
    "overlay",
    "policy",
    "readme",
    "license",
)
ACTION_WORDS = re.compile(
    r"\b(enable|disable|turn on|turn off|open|tap|select|choose|configure|"
    r"allow|deny|reset|manage)\b",
    re.IGNORECASE,
)
SETTING_SIGNALS = re.compile(
    r"\b(setting|settings|preference|preferences|permission|permissions|toggle|switch|"
    r"privacy|security|developer options?|network|location|notification)\b",
    re.IGNORECASE,
)
UI_PATH_PATTERN = re.compile(
    r"(?:go to|open|navigate to)?\s*(?:the\s+)?settings\s*(?:app\s*)?"
    r"(?:>|→|›|/|\\|then|and then|:)\s*([^\n.;]{2,240})",
    re.IGNORECASE,
)
SPLIT_PATH_PATTERN = re.compile(r"\s*(?:>|→|›|/|\\|\bthen\b|\band then\b)\s*", re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")
SLUG_CHARS = re.compile(r"[^a-z0-9]+")


class ScrapeError(RuntimeError):
    """Base error for snapshot or normalization failures."""


class SourceUnavailableError(ScrapeError):
    """Raised when a configured source cannot be acquired."""


class AllowlistError(ScrapeError):
    """Raised when a URL escapes the configured source boundary."""


class StaleSourceError(ScrapeError):
    """Raised when a source revision changed under a stale-source gate."""


def normalize_repository_subpath(value: object) -> str | None:
    """Return a safe repository-relative POSIX directory path."""
    if value is None:
        return None
    raw = str(value).strip().replace("\\", "/")
    if not raw:
        return None
    parts = raw.split("/")
    if raw.startswith("/") or "\x00" in raw or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("repository_subpath must be a normalized repository-relative path")
    return "/".join(parts)


@dataclass(frozen=True)
class SourceSpec:
    source_key: str
    os_family: str
    root_url: str
    repository_url: str | None
    repository_subpath: str | None
    confidence_tier: int
    license_identifier: str
    discovery_mode: str
    allowed_hosts: tuple[str, ...]
    required_classes: tuple[str, ...]
    enabled: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "SourceSpec":
        allowed_hosts = tuple(str(value).lower() for value in raw.get("allowed_hosts") or ())
        required_classes = tuple(str(value) for value in raw.get("required_classes") or ())
        spec = cls(
            source_key=str(raw.get("source_key") or ""),
            os_family=str(raw.get("os_family") or ""),
            root_url=str(raw.get("root_url") or ""),
            repository_url=str(raw["repository_url"]) if raw.get("repository_url") else None,
            repository_subpath=normalize_repository_subpath(raw.get("repository_subpath")),
            confidence_tier=int(raw.get("confidence_tier", 1)),
            license_identifier=str(raw.get("license_identifier") or ""),
            discovery_mode=str(raw.get("discovery_mode") or "official_index"),
            allowed_hosts=allowed_hosts,
            required_classes=required_classes,
            enabled=bool(raw.get("enabled", True)),
        )
        if not spec.source_key or not spec.os_family or not spec.root_url:
            raise ValueError("source_key, os_family and root_url are required")
        if not spec.allowed_hosts:
            raise ValueError(f"source {spec.source_key} needs allowed_hosts")
        if not spec.license_identifier:
            raise ValueError(f"source {spec.source_key} needs license_identifier")
        if not 0 <= spec.confidence_tier <= 6:
            raise ValueError(f"source {spec.source_key} has invalid confidence_tier")
        validate_url(spec.root_url, spec.allowed_hosts)
        if spec.repository_url:
            validate_url(spec.repository_url, spec.allowed_hosts)
        return spec


@dataclass(frozen=True)
class CrawlLimits:
    max_pages_per_source: int = 250
    max_depth: int = 4
    request_timeout: float = 20.0
    delay_seconds: float = 0.25
    max_bytes: int = 4 * 1024 * 1024
    user_agent: str = DEFAULT_USER_AGENT

    def __post_init__(self) -> None:
        if min(self.max_pages_per_source, self.max_bytes) < 1:
            raise ValueError("page and byte limits must be positive")
        if self.max_depth < 0 or self.request_timeout <= 0 or self.delay_seconds < 0:
            raise ValueError("invalid crawl limits")


@dataclass(frozen=True)
class RawDocument:
    source_key: str
    os_family: str
    canonical_url: str
    body: bytes
    media_type: str
    source_revision: str | None = None
    source_snapshot_sha256: str | None = None
    relative_path: str | None = None

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.body).hexdigest()


@dataclass(frozen=True)
class TextBlock:
    heading_path: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class SourceResult:
    source: SourceSpec
    revision: str
    documents: tuple[dict[str, object], ...]
    observed_classes: tuple[str, ...]
    snapshot_paths: tuple[str, ...]
    warnings: tuple[str, ...]


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.blocks: list[TextBlock] = []
        self._heading_stack: list[str] = []
        self._capture_tag: str | None = None
        self._capture_level: int | None = None
        self._capture: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "svg", "noscript", "template"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if lowered == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        if re.fullmatch(r"h[1-6]", lowered):
            self._start_capture(lowered, int(lowered[1]))
        elif lowered in {"p", "li", "pre", "code", "dt", "dd", "td", "th"}:
            self._start_capture(lowered, None)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "svg", "noscript", "template"}:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth or lowered != self._capture_tag:
            return
        text = clean_text(" ".join(self._capture))
        if text:
            if self._capture_level is not None:
                level = self._capture_level
                self._heading_stack = self._heading_stack[: level - 1]
                while len(self._heading_stack) < level - 1:
                    self._heading_stack.append("")
                self._heading_stack.append(text)
            else:
                path = tuple(item for item in self._heading_stack if item)
                self.blocks.append(TextBlock(path, text))
        self._capture_tag = None
        self._capture_level = None
        self._capture = []

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and self._capture_tag is not None:
            self._capture.append(data)

    def _start_capture(self, tag: str, level: int | None) -> None:
        if self._capture_tag is None:
            self._capture_tag = tag
            self._capture_level = level
            self._capture = []


def clean_text(value: str) -> str:
    return WHITESPACE.sub(" ", html.unescape(value)).strip()


def validate_url(url: str, allowed_hosts: Sequence[str]) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"https", "http"}:
        raise AllowlistError(f"unsupported URL scheme: {url}")
    if parsed.scheme == "http" and host not in {"localhost", "127.0.0.1", "::1"}:
        raise AllowlistError(f"non-loopback HTTP source is forbidden: {url}")
    if not host_allowed(host, allowed_hosts):
        raise AllowlistError(f"URL host is not allowlisted: {url}")
    if parsed.username or parsed.password:
        raise AllowlistError(f"credentials in source URL are forbidden: {url}")
    return url


def host_allowed(host: str, allowed_hosts: Sequence[str]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == value or host.endswith("." + value) for value in allowed_hosts)


def canonicalize_url(url: str) -> str:
    clean, _ = urldefrag(url)
    parsed = urlparse(clean)
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", parsed.query, ""))


class HttpCrawler:
    def __init__(self, limits: CrawlLimits):
        self.limits = limits
        self.opener = build_opener()
        self._last_request: dict[str, float] = {}
        self._robots: dict[tuple[str, str], RobotFileParser] = {}

    def crawl(self, source: SourceSpec) -> tuple[tuple[RawDocument, ...], tuple[str, ...]]:
        root = canonicalize_url(validate_url(source.root_url, source.allowed_hosts))
        queue: deque[tuple[str, int]] = deque([(root, 0)])
        seen: set[str] = set()
        documents: list[RawDocument] = []
        warnings: list[str] = []
        while queue and len(documents) < self.limits.max_pages_per_source:
            url, depth = queue.popleft()
            if url in seen:
                continue
            seen.add(url)
            try:
                raw, final_url, media_type = self._fetch(url, source)
            except ScrapeError as exc:
                if url == root:
                    raise SourceUnavailableError(f"cannot fetch root for {source.source_key}: {exc}") from exc
                warnings.append(f"{url}: {exc}")
                continue
            document = RawDocument(source.source_key, source.os_family, final_url, raw, media_type)
            documents.append(document)
            if depth >= self.limits.max_depth or "html" not in media_type:
                continue
            parser = parse_html(raw)
            for href in parser.links:
                candidate = canonicalize_url(urljoin(final_url, href))
                if not self._candidate_allowed(candidate, source, root):
                    continue
                if candidate not in seen:
                    queue.append((candidate, depth + 1))
            if source.discovery_mode == "single_versioned_page":
                break
        if not documents:
            raise SourceUnavailableError(f"source {source.source_key} produced no documents")
        return tuple(documents), tuple(warnings)

    def _fetch(self, url: str, source: SourceSpec) -> tuple[bytes, str, str]:
        validate_url(url, source.allowed_hosts)
        if not self._robots_allowed(url, source):
            raise ScrapeError("robots.txt disallows this URL")
        host = (urlparse(url).hostname or "").lower()
        elapsed = time.monotonic() - self._last_request.get(host, 0.0)
        if elapsed < self.limits.delay_seconds:
            time.sleep(self.limits.delay_seconds - elapsed)
        request = Request(
            url,
            headers={
                "User-Agent": self.limits.user_agent,
                "Accept": "text/html,text/plain,text/markdown,application/json,application/xml;q=0.9,*/*;q=0.1",
            },
        )
        try:
            with self.opener.open(request, timeout=self.limits.request_timeout) as response:
                final_url = canonicalize_url(response.geturl())
                validate_url(final_url, source.allowed_hosts)
                media_type = response.headers.get_content_type().lower()
                if not any(media_type.startswith(item) for item in TEXT_CONTENT_TYPES):
                    raise ScrapeError(f"unsupported content type: {media_type}")
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > self.limits.max_bytes:
                    raise ScrapeError("response exceeds max byte limit")
                raw = response.read(self.limits.max_bytes + 1)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            raise ScrapeError(str(exc)) from exc
        finally:
            self._last_request[host] = time.monotonic()
        if len(raw) > self.limits.max_bytes:
            raise ScrapeError("response exceeds max byte limit")
        return raw, final_url, media_type

    def _robots_allowed(self, url: str, source: SourceSpec) -> bool:
        parsed = urlparse(url)
        key = (parsed.scheme, parsed.netloc)
        parser = self._robots.get(key)
        if parser is None:
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            parser = RobotFileParser(robots_url)
            try:
                request = Request(robots_url, headers={"User-Agent": self.limits.user_agent})
                with self.opener.open(request, timeout=self.limits.request_timeout) as response:
                    final_url = response.geturl()
                    validate_url(final_url, source.allowed_hosts)
                    content = response.read(min(self.limits.max_bytes, 512 * 1024)).decode("utf-8", errors="replace")
                parser.parse(content.splitlines())
            except HTTPError as exc:
                if exc.code in {401, 403}:
                    parser.parse(["User-agent: *", "Disallow: /"])
                else:
                    parser.parse(["User-agent: *", "Disallow:"])
            except (URLError, TimeoutError, OSError, AllowlistError):
                parser.parse(["User-agent: *", "Disallow:"])
            self._robots[key] = parser
        return parser.can_fetch(self.limits.user_agent, url)

    @staticmethod
    def _candidate_allowed(candidate: str, source: SourceSpec, root: str) -> bool:
        try:
            validate_url(candidate, source.allowed_hosts)
        except AllowlistError:
            return False
        parsed = urlparse(candidate)
        root_parsed = urlparse(root)
        if parsed.path.lower().endswith((".apk", ".bin", ".gz", ".iso", ".jar", ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".zip")):
            return False
        if any(token in parsed.path.lower() for token in ("/login", "/signup", "/search", "/feed", "/archive")):
            return False
        if parsed.netloc == root_parsed.netloc:
            root_prefix = root_parsed.path.rsplit("/", 1)[0]
            return parsed.path.startswith(root_prefix or "/")
        return True


class GitSnapshotter:
    def __init__(self, limits: CrawlLimits):
        self.limits = limits

    def can_snapshot(self, source: SourceSpec) -> bool:
        url = source.repository_url or ""
        path = urlparse(url).path.rstrip("/")
        return bool(url and path and path.count("/") >= 2 and shutil.which("git"))

    def snapshot(self, source: SourceSpec) -> tuple[str, tuple[RawDocument, ...], tuple[str, ...]]:
        repository = source.repository_url or ""
        validate_url(repository, source.allowed_hosts)
        revision = self._resolve_head(repository)
        warnings: list[str] = []
        with tempfile.TemporaryDirectory(prefix="os-settings-git-") as temporary:
            checkout = Path(temporary) / "repo"
            command = [
                "git",
                "-c",
                "http.followRedirects=false",
                "clone",
                "--quiet",
                "--depth",
                "1",
                "--filter=blob:none",
                "--no-tags",
            ]
            if source.repository_subpath:
                command.append("--no-checkout")
            command.extend((repository, str(checkout)))
            self._run(command, timeout=max(60.0, self.limits.request_timeout * 3))
            if source.repository_subpath:
                self._run(["git", "-C", str(checkout), "sparse-checkout", "init", "--cone"])
                self._run(
                    [
                        "git",
                        "-C",
                        str(checkout),
                        "sparse-checkout",
                        "set",
                        "--cone",
                        source.repository_subpath,
                    ]
                )
                self._run(
                    ["git", "-C", str(checkout), "checkout", "--quiet", "--detach", revision],
                    timeout=max(60.0, self.limits.request_timeout * 3),
                )
            actual = self._run(["git", "-C", str(checkout), "rev-parse", "HEAD"]).strip()
            if actual != revision:
                revision = actual
            scan_root = checkout / source.repository_subpath if source.repository_subpath else checkout
            if not scan_root.is_dir():
                raise SourceUnavailableError(
                    f"repository subpath does not exist at {revision}: {source.repository_subpath}"
                )
            candidates = [path for path in scan_root.rglob("*") if self._eligible(path, checkout)]
            candidates.sort(key=lambda path: (not self._relevant(path, checkout), str(path.relative_to(checkout))))
            documents: list[RawDocument] = []
            for path in candidates[: self.limits.max_pages_per_source]:
                try:
                    raw = path.read_bytes()
                except OSError as exc:
                    warnings.append(f"{path}: {exc}")
                    continue
                if not raw or len(raw) > self.limits.max_bytes or b"\x00" in raw[:4096]:
                    continue
                relative = path.relative_to(checkout).as_posix()
                canonical = f"git+{repository}@{revision}#{relative}"
                documents.append(
                    RawDocument(
                        source.source_key,
                        source.os_family,
                        canonical,
                        raw,
                        media_type_for_path(path),
                        source_revision=revision,
                        relative_path=relative,
                    )
                )
        if not documents:
            raise SourceUnavailableError(f"git source {source.source_key} produced no text documents")
        return revision, tuple(documents), tuple(warnings)

    def _resolve_head(self, repository: str) -> str:
        output = self._run(["git", "-c", "http.followRedirects=false", "ls-remote", repository, "HEAD"])
        revision = output.split()[0] if output.split() else ""
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
            raise SourceUnavailableError(f"cannot resolve immutable HEAD for {repository}")
        return revision.lower()

    @staticmethod
    def _run(command: Sequence[str], timeout: float = 30.0) -> str:
        try:
            environment = dict(os.environ)
            environment.update({"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never", "LC_ALL": "C"})
            completed = subprocess.run(
                list(command),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                env=environment,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            raise SourceUnavailableError(clean_text(str(detail))[:500]) from exc
        return completed.stdout

    @staticmethod
    def _eligible(path: Path, checkout: Path) -> bool:
        if not path.is_file() or ".git" in path.relative_to(checkout).parts:
            return False
        return path.suffix.lower() in TEXT_EXTENSIONS or path.name.lower() in {"readme", "license", "notice"}

    @staticmethod
    def _relevant(path: Path, checkout: Path) -> bool:
        value = path.relative_to(checkout).as_posix().lower()
        return any(token in value for token in RELEVANT_PATH_PARTS)


def media_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm"}:
        return "text/html"
    if suffix == ".json":
        return "application/json"
    if suffix == ".xml":
        return "application/xml"
    if suffix == ".md":
        return "text/markdown"
    return "text/plain"


def parse_html(raw: bytes) -> _PageParser:
    parser = _PageParser()
    parser.feed(raw.decode("utf-8", errors="replace"))
    parser.close()
    return parser


def text_blocks(document: RawDocument) -> tuple[TextBlock, ...]:
    text = document.body.decode("utf-8", errors="replace")
    if "html" in document.media_type:
        parsed = parse_html(document.body)
        if parsed.blocks:
            return tuple(parsed.blocks)
    blocks: list[TextBlock] = []
    heading_path: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        value = clean_text(" ".join(paragraph))
        paragraph.clear()
        if value:
            blocks.append(TextBlock(tuple(heading_path), value))

    for line in text.splitlines():
        stripped = line.strip()
        markdown = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if markdown:
            flush()
            level = len(markdown.group(1))
            heading_path[:] = heading_path[: level - 1]
            while len(heading_path) < level - 1:
                heading_path.append("")
            heading_path.append(clean_text(markdown.group(2)))
        elif not stripped:
            flush()
        else:
            paragraph.append(stripped)
    flush()
    if not blocks and clean_text(text):
        blocks.append(TextBlock((), clean_text(text)))
    return tuple(blocks)


def infer_document_classes(document: RawDocument, blocks: Sequence[TextBlock]) -> tuple[str, ...]:
    canonical = document.canonical_url.lower()
    relative = (document.relative_path or "").lower().replace("\\", "/")
    sample = " ".join((canonical, relative, *(block.text[:800] for block in blocks[:40]))).lower()
    classes: set[str] = set()

    android_settings_contract = any(
        token in sample
        for token in (
            "/reference/android/provider/settings",
            "android.provider.settings",
            "provider/settings",
            "settings.provider",
        )
    )
    if any(token in sample for token in ("/reference/", " api ")) or android_settings_contract:
        classes.add("api_reference")

    # Android's Settings provider/API is itself a permission and security
    # boundary even when the rendered reference shell omits permission prose.
    if android_settings_contract or any(
        token in sample
        for token in (
            "permission",
            "requires permission",
            "security",
            "privacy",
            "encryption",
            "app op",
            "write_settings",
            "read_device_config",
        )
    ):
        classes.add("permission_security")

    if document.canonical_url.startswith("git+") or (
        document.relative_path
        and Path(document.relative_path).suffix.lower() in {".kt", ".java", ".c", ".cc", ".cpp", ".h", ".py", ".xml"}
    ):
        classes.add("source_code")

    settings_resource = bool(
        relative
        and (
            re.search(r"(^|/)(res|res-product|res-export)/(xml|values|menu|navigation)(/|$)", relative)
            or "/overlay" in relative
            or relative.startswith("overlay")
            or relative.endswith(("preferences.xml", "settings.xml", "config.xml"))
        )
    )
    if any(token in sample for token in ("setting", "preference", "toggle", "switch")):
        classes.add("settings_ui_path")
    if settings_resource or any(
        token in sample
        for token in ("resource overlay", "runtime resource overlay", " rro ", "res/values", "res/xml", "config.xml")
    ):
        classes.add("resource_overlay")

    if any(token in sample for token in ("guide", "how-to", "how to", "documentation", "docs/")):
        classes.add("user_guide")
    if any(token in sample for token in ("release note", "changelog", "what's new", "whats-new")):
        classes.add("release_notes")

    emulator_support = any(
        token in sample
        for token in (
            "/how-to/emulator",
            " emulator image",
            "android emulator",
            "virtual device",
            "avd manager",
            "create virtual device",
            "system-images/android-",
            "system image",
            "android api",
            "nexus 5x",
            "sdk_phone",
        )
    )
    if emulator_support or any(
        token in sample
        for token in ("supported device", "device support", "compatibility", "hardware requirements")
    ):
        classes.add("device_support")
    if any(token in sample for token in ("license", "copyright", "legal", "cc by-sa", "apache license")):
        classes.add("legal_license")
    if any(token in sample for token in ("enterprise", "device owner", "administrator", "policy")):
        classes.add("enterprise_policy")
    return tuple(sorted(classes or {"informational"}))


def normalize_document(source: SourceSpec, document: RawDocument) -> dict[str, object]:
    blocks = text_blocks(document)
    classes = infer_document_classes(document, blocks)
    claims: list[dict[str, object]] = []
    sections: list[dict[str, object]] = []
    groups: dict[tuple[str, ...], list[TextBlock]] = {}
    for block in blocks:
        groups.setdefault(block.heading_path, []).append(block)
    if not groups:
        groups[()] = []
    for ordinal, (heading_path, grouped_blocks) in enumerate(groups.items()):
        combined = clean_text(" ".join(block.text for block in grouped_blocks))
        heading = next((value for value in reversed(heading_path) if value), "")
        section_id = stable_id(document.canonical_url, "/".join(heading_path), str(ordinal))
        claim_indexes: list[int] = []
        path = extract_ui_path(combined)
        # A setting_variant is UI-groundable only when the source contains an explicit path.
        # Descriptive settings prose without a path remains evidence, but is not emitted
        # as a claim that downstream curriculum code could mistake for a visible target.
        actionable = bool(path) and bool(heading or combined)
        if actionable:
            display_name = clean_text(heading or (path[-1] if path else combined[:100]))[:160]
            if display_name:
                claim_indexes.append(len(claims))
                claims.append(
                    {
                        "claim_kind": "setting_variant",
                        "canonical_setting_hint": f"{source.os_family}.{slugify(display_name)}",
                        "display_name": display_name,
                        "ui_path": path,
                        "control_plane": "settings_ui",
                        "required_role": infer_required_role(combined),
                        "risk_level": infer_risk_level(combined),
                        "scope": {
                            "os_family": source.os_family,
                            "source_key": source.source_key,
                            "canonical_url": document.canonical_url,
                            "section_id": section_id,
                        },
                    }
                )
        sections.append(
            {
                "section_id": section_id,
                "document_class": choose_section_class(classes, combined),
                "heading_path": list(heading_path),
                "source_locator": {
                    "canonical_url": document.canonical_url,
                    "relative_path": document.relative_path,
                    "block_ordinal": ordinal,
                },
                "claim_indexes": claim_indexes,
            }
        )
    document_id = stable_id(source.source_key, document.canonical_url)
    return {
        "document": {
            "source_id": source.source_key,
            "document_id": document_id,
            "canonical_url": document.canonical_url,
            "snapshot_sha256": document.source_snapshot_sha256 or document.content_sha256,
            "content_sha256": document.content_sha256,
            "source_revision": document.source_revision,
            "confidence_tier": source.confidence_tier,
            "license_identifier": source.license_identifier,
            "document_type": list(classes),
            "os_family": source.os_family,
        },
        "claims": claims,
        "sections": sections,
    }


def extract_ui_path(text: str) -> list[str]:
    match = UI_PATH_PATTERN.search(text)
    if not match:
        return []
    tail = match.group(1)
    values = [clean_text(value.strip(" .,:;()[]")) for value in SPLIT_PATH_PATTERN.split(tail)]
    values = [value for value in values if 1 < len(value) <= 100]
    return ["Settings", *values[:7]] if values else []


def infer_required_role(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("root access", "root permission", "su ")):
        return "root"
    if any(token in lowered for token in ("device owner", "enterprise administrator", "admin only")):
        return "device_owner"
    if any(token in lowered for token in ("developer options", "adb", "debugging")):
        return "developer"
    return "user"


def infer_risk_level(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("factory reset", "erase", "wipe", "unlock bootloader", "root", "encryption key")):
        return "high"
    if any(token in lowered for token in ("permission", "security", "privacy", "location", "network", "vpn", "developer options", "debugging")):
        return "medium"
    return "low"


def choose_section_class(classes: Sequence[str], text: str) -> str:
    inferred = infer_document_classes(
        RawDocument("", "", "section", text.encode("utf-8"), "text/plain"),
        (TextBlock((), text),),
    )
    for value in inferred:
        if value in classes and value != "informational":
            return value
    return next((value for value in classes if value != "informational"), "informational")


def slugify(value: str) -> str:
    slug = SLUG_CHARS.sub("_", value.casefold()).strip("_")
    return slug[:96] or "unknown_setting"


def stable_id(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()[:24]


def write_snapshot(snapshot_dir: Path, document: RawDocument) -> str:
    extension = {
        "text/html": ".html",
        "text/markdown": ".md",
        "application/json": ".json",
        "application/xml": ".xml",
    }.get(document.media_type, ".txt")
    relative = Path("raw") / document.source_key / f"{document.content_sha256}{extension}"
    target = snapshot_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != document.content_sha256:
            raise ScrapeError(f"snapshot collision at {target}")
    else:
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(document.body)
        os.replace(temporary, target)
    return relative.as_posix()


def acquire_source(source: SourceSpec, limits: CrawlLimits, snapshot_dir: Path) -> SourceResult:
    warnings: tuple[str, ...]
    if source.discovery_mode in {"git_commit", "pinned_manifest"} and GitSnapshotter(limits).can_snapshot(source):
        revision, raw_documents, warnings = GitSnapshotter(limits).snapshot(source)
    else:
        raw_documents, warnings = HttpCrawler(limits).crawl(source)
        revision_material = "\n".join(
            f"{item.canonical_url}\0{item.content_sha256}" for item in sorted(raw_documents, key=lambda value: value.canonical_url)
        )
        revision = hashlib.sha256(revision_material.encode("utf-8")).hexdigest()
    source_snapshot_sha256 = hashlib.sha256(f"{source.source_key}\0{revision}".encode("utf-8")).hexdigest()
    normalized: list[dict[str, object]] = []
    snapshot_paths: list[str] = []
    observed_classes: set[str] = set()
    seen_snapshot_urls: set[tuple[str, str]] = set()
    for raw in raw_documents:
        raw = RawDocument(
            source_key=raw.source_key,
            os_family=raw.os_family,
            canonical_url=raw.canonical_url,
            body=raw.body,
            media_type=raw.media_type,
            source_revision=revision,
            source_snapshot_sha256=source_snapshot_sha256,
            relative_path=raw.relative_path,
        )
        key = (raw.canonical_url, raw.content_sha256)
        if key in seen_snapshot_urls:
            continue
        seen_snapshot_urls.add(key)
        snapshot_paths.append(write_snapshot(snapshot_dir, raw))
        value = normalize_document(source, raw)
        observed_classes.update(str(item) for item in value["document"]["document_type"])
        normalized.append(value)
    normalized.sort(key=lambda item: (str(item["document"]["canonical_url"]), str(item["document"]["document_id"])))
    return SourceResult(
        source=source,
        revision=revision,
        documents=tuple(normalized),
        observed_classes=tuple(sorted(observed_classes)),
        snapshot_paths=tuple(snapshot_paths),
        warnings=warnings,
    )


def load_sources(path: str | Path) -> tuple[SourceSpec, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("sources") if isinstance(payload, Mapping) else payload
    if not isinstance(values, list):
        raise ValueError("source registry must contain a sources list")
    return tuple(SourceSpec.from_mapping(item) for item in values if isinstance(item, Mapping))


def select_sources(
    sources: Sequence[SourceSpec],
    requested_os: Sequence[str],
    all_enabled_os: bool,
) -> tuple[SourceSpec, ...]:
    enabled = tuple(source for source in sources if source.enabled)
    if all_enabled_os and requested_os:
        raise ValueError("--all-enabled-os cannot be combined with --os-family")
    requested = set(requested_os)
    if requested:
        unknown = requested - {source.os_family for source in enabled}
        if unknown:
            raise ValueError(f"unknown or disabled OS families: {sorted(unknown)}")
        return tuple(source for source in enabled if source.os_family in requested)
    return enabled


def previous_revisions(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    values = payload.get("sources") if isinstance(payload, Mapping) else None
    if not isinstance(values, list):
        raise ValueError(f"invalid prior manifest: {manifest_path}")
    return {
        str(item.get("source_key")): str(item.get("revision"))
        for item in values
        if isinstance(item, Mapping) and item.get("source_key") and item.get("revision")
    }


def validate_results(
    selected: Sequence[SourceSpec],
    results: Sequence[SourceResult],
    *,
    require_all_os: bool,
    require_required_classes: bool,
) -> None:
    by_os: dict[str, list[dict[str, object]]] = {}
    observed_by_source = {result.source.source_key: set(result.observed_classes) for result in results}
    for result in results:
        by_os.setdefault(result.source.os_family, []).extend(result.documents)
    if require_all_os:
        failures: list[str] = []
        for os_family in sorted({source.os_family for source in selected}):
            documents = by_os.get(os_family) or []
            claims = sum(len(document.get("claims") or ()) for document in documents)
            if not documents:
                failures.append(f"{os_family}: no normalized documents")
            elif not claims:
                failures.append(f"{os_family}: no setting_variant claims")
        if failures:
            raise ScrapeError("required OS coverage failed: " + "; ".join(failures))
    if require_required_classes:
        missing: dict[str, list[str]] = {}
        for source in selected:
            absent = sorted(set(source.required_classes) - observed_by_source.get(source.source_key, set()))
            if absent:
                missing[source.source_key] = absent
        if missing:
            raise ScrapeError(f"required document classes are missing: {missing}")


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, object]:
    sources = load_sources(args.sources)
    selected = select_sources(sources, args.os_family, args.all_enabled_os)
    if not selected:
        raise ScrapeError("no enabled sources selected")
    over_tier = [source.source_key for source in selected if source.confidence_tier > args.max_evidence_tier]
    if over_tier:
        raise ScrapeError(
            f"selected sources exceed --max-evidence-tier={args.max_evidence_tier}: {over_tier}"
        )
    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = snapshot_dir / "manifest.json"
    prior = previous_revisions(manifest_path)
    limits = CrawlLimits(
        max_pages_per_source=args.max_pages_per_source,
        max_depth=args.max_depth,
        request_timeout=args.request_timeout,
        delay_seconds=args.delay_seconds,
        max_bytes=args.max_bytes,
        user_agent=args.user_agent,
    )
    results = [acquire_source(source, limits, snapshot_dir) for source in selected]
    stale = {
        result.source.source_key: {"previous": prior[result.source.source_key], "current": result.revision}
        for result in results
        if result.source.source_key in prior and prior[result.source.source_key] != result.revision
    }
    if stale and args.fail_on_stale_source:
        raise StaleSourceError(f"source revisions changed: {stale}")
    validate_results(
        selected,
        results,
        require_all_os=args.require_all_os,
        require_required_classes=not args.allow_missing_required_classes,
    )
    documents = [document for result in results for document in result.documents]
    documents.sort(
        key=lambda item: (
            str(item["document"]["os_family"]),
            str(item["document"]["source_id"]),
            str(item["document"]["canonical_url"]),
        )
    )
    output = Path(args.output)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output": output.as_posix(),
        "document_count": len(documents),
        "setting_claim_count": sum(len(item.get("claims") or ()) for item in documents),
        "stale_sources": stale,
        "sources": [
            {
                "source_key": result.source.source_key,
                "os_family": result.source.os_family,
                "discovery_mode": result.source.discovery_mode,
                "revision": result.revision,
                "document_count": len(result.documents),
                "setting_claim_count": sum(len(item.get("claims") or ()) for item in result.documents),
                "required_classes": list(result.source.required_classes),
                "observed_classes": list(result.observed_classes),
                "snapshot_paths": list(result.snapshot_paths),
                "warnings": list(result.warnings),
            }
            for result in results
        ],
    }
    atomic_write_json(output, documents)
    atomic_write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="config/document_sources.json")
    parser.add_argument("--os-family", action="append", default=[], help="OS family to include; repeatable")
    parser.add_argument("--all-enabled-os", action="store_true", help="include every enabled OS family")
    parser.add_argument("--snapshot-dir", default="artifacts/document-snapshots")
    parser.add_argument("--output", default="artifacts/normalized-documents.json")
    parser.add_argument("--max-evidence-tier", type=int, default=2, help="reject sources above this confidence tier")
    parser.add_argument("--require-all-os", action="store_true", help="require documents and setting claims for every selected OS")
    parser.add_argument("--fail-on-stale-source", action="store_true", help="reject revisions that differ from the prior manifest")
    parser.add_argument("--allow-missing-required-classes", action="store_true", help="do not fail when a source misses a registry-required document class")
    parser.add_argument("--max-pages-per-source", type=int, default=250)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=20.0)
    parser.add_argument("--delay-seconds", type=float, default=0.25)
    parser.add_argument("--max-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = run(args)
    except (ScrapeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
