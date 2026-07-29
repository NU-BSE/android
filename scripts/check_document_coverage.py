#!/usr/bin/env python3
"""Fail CI/release gating when a coverage ledger is incomplete or stale."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger")
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()
    payload = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
    report = payload.get("report") or {}
    total = int(report.get("total_sections", 0))
    terminal = int(report.get("terminal_sections", 0))
    blocked = int(report.get("blocked_sections", 0))
    missing = report.get("missing_required_classes") or {}
    stale = report.get("stale_sources") or ()
    failures = []
    if total <= 0 or terminal != total:
        failures.append(f"terminal coverage {terminal}/{total}")
    if blocked and not args.allow_blocked:
        failures.append(f"{blocked} blocked sections")
    if missing:
        failures.append(f"missing document classes: {missing}")
    if stale:
        failures.append(f"stale source revisions: {stale}")
    if failures:
        raise SystemExit("coverage gate failed: " + "; ".join(failures))
    print(f"coverage complete: {terminal}/{total}")


if __name__ == "__main__":
    main()
