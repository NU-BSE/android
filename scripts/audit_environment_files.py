"""Audit ARC-AGI-3 environment packages for DRLQN submission compatibility.

Usage:
    python scripts/audit_environment_files.py /path/to/environment_files

The audit is intentionally static: it scans Python sources for declared
GameAction ids and flags requirements outside ACTION1..ACTION6/RESET. It also
reports click-heavy games where dynamic spatial proposals are essential.
"""
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import re

SUPPORTED = set(range(0, 7))
ACTION_RE = re.compile(r"(?:GameAction\.)?ACTION(\d+)")


@dataclass
class EnvironmentAudit:
    game: str
    source: str
    action_ids: list[int]
    unsupported: list[int]
    uses_click: bool
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
                # Action declarations typically contain a short subset of 1..7.
                if any(1 <= value <= 7 for value in values) and len(values) <= 10:
                    ids.update(values)
    return ids


def audit_file(path: Path) -> EnvironmentAudit:
    text = path.read_text(encoding="utf-8", errors="replace")
    action_ids = {int(match.group(1)) for match in ACTION_RE.finditer(text)}
    try:
        action_ids.update(_literal_action_lists(ast.parse(text)))
    except SyntaxError:
        pass
    # Remove common non-action constants accidentally captured by literal-list
    # scanning unless they also appeared as explicit ACTIONn names.
    explicit = {int(match.group(1)) for match in ACTION_RE.finditer(text)}
    if explicit:
        action_ids = {value for value in action_ids if value in explicit or value <= 7}
    unsupported = sorted(action_ids - SUPPORTED)
    uses_click = 6 in action_ids or "ACTION6" in text
    game = path.stem
    status = "unsupported-actions" if unsupported else "compatible"
    return EnvironmentAudit(
        game=game,
        source=str(path),
        action_ids=sorted(action_ids),
        unsupported=unsupported,
        uses_click=uses_click,
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
    summary = {
        "files": len(results),
        "click_files": len(click_games),
        "unsupported_files": len(unsupported),
        "unsupported": [asdict(item) for item in unsupported],
    }
    print(json.dumps(summary, indent=2))
    if args.json_path:
        args.json_path.write_text(
            json.dumps([asdict(item) for item in results], indent=2),
            encoding="utf-8",
        )
    return 1 if unsupported else 0


if __name__ == "__main__":
    raise SystemExit(main())
