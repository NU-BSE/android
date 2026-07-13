"""Merge resumable DRLQN matrix JSONL shards and recompute aggregates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_drlqn_matrix import _aggregate, _run_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--rows", type=Path, default=Path("results/drlqn_baseline_matrix.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/drlqn_baseline_matrix.json"))
    args = parser.parse_args()
    unique = {}
    for shard in args.shards:
        paths = sorted(shard.rglob("*.jsonl")) if shard.is_dir() else [shard]
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                unique[_run_key(row)] = row
    rows = [unique[key] for key in sorted(unique, key=lambda key: tuple(map(str, key)))]
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    args.rows.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    payload = {
        "runs": len(rows),
        "aggregate": _aggregate(rows),
        "errors": [row for row in rows if row.get("error")],
        "rows_file": str(args.rows),
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"runs": len(rows), "errors": len(payload["errors"])}, sort_keys=True))


if __name__ == "__main__":
    main()
