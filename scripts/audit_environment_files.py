"""Audit ARC-AGI-3 environment packages for DRLQN submission compatibility.

Usage:
    python scripts/audit_environment_files.py /path/to/environment_files

The audit is intentionally static: it scans Python sources for declared
GameAction ids, reports click/extra-key games, and flags requirements outside
the fixed DRLQN mapping ACTION1..ACTION9/RESET.
"""
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import re

SUPPORTED = set(range(0, 10))
ACTION_RE = re.compile(r"(?:GameAction\.)?ACTION(\d+)")


@dataclass
class EnvironmentAudit:
    game: str
    source: str
    action_ids: list[int]
    unsupported: list[int]
    uses_click: bool
    uses_extra_key: bool
    status: str


def _literal_action_lists(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values: list[int] = []
            ok = True
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    values.append(int(elt.value))
                else:
                    ok = False
                    break
            if ok and values and all(0 <= value <= 20 for value in values):
                # Action declarations contain a short subset of engine ids.
                if any(1 <= value <= 9 for value in values) and len(values) <= 12:
                    ids.update(values)
    return ids


def audit_file(path: Path) -> EnvironmentAudit:
    text = path.read_text(encoding="utf-8", errors="replace")
    explicit = {int(match.group(1)) for match in ACTION_RE.finditer(text)}
    action_ids = set(explicit)
    try:
        action_ids.update(_literal_action_lists(ast.parse(text)))
    except SyntaxError:
        pass
    # Remove common non-action constants accidentally captured by literal-list
    # scanning unless they also appeared as explicit ACTIONn names.
    if explicit:
        action_ids = {value for value in action_ids if value in explicit or value <= 9}
    unsupported = sorted(action_ids - SUPPORTED)
    uses_click = 6 in action_ids or "ACTION6" in text
    uses_extra_key = any(value >= 7 for value in action_ids)
    game = path.stem
    status = "unsupported-actions" if unsupported else "compatible"
    return EnvironmentAudit(
        game=game,
        source=str(path),
        action_ids=sorted(action_ids),
        unsupported=unsupported,
        uses_click=uses_click,
        uses_extra_key=uses_extra_key,
        status=status,
    )


def audit_tree(root: Path) -> list[EnvironmentAudit]:
    results: list[EnvironmentAudit] = []
    for path in sorted(root.rglob("*.py")):
        if path.name.startswith("_"):
            continue
        results.append(audit_file(path))
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args()
    results = audit_tree(args.root)
    unsupported = [item for item in results if item.unsupported]
    click_games = [item for item in results if item.uses_click]
    extra_key_games = [item for item in results if item.uses_extra_key]
    summary = {
        "files": len(results),
        "click_files": len(click_games),
        "extra_key_files": len(extra_key_games),
        "extra_key_games": [item.game for item in extra_key_games],
        "max_action_id": max(
            (max(item.action_ids, default=0) for item in results), default=0
        ),
        "unsupported_files": len(unsupported),
        "unsupported": [asdict(item) for item in unsupported],
    }
    print(json.dumps(summary, indent=2))
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps([asdict(item) for item in results], indent=2),
            encoding="utf-8",
        )
    return 1 if unsupported else 0


if __name__ == "__main__":
    raise SystemExit(main())
