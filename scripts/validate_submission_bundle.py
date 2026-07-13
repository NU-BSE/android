"""Validate bundle hashes, offline wheels and checkpoint loading in isolation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--install-wheels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.bundle.exists():
        raise SystemExit(f"bundle does not exist: {args.bundle}")
    with tempfile.TemporaryDirectory(prefix="drlqn-bundle-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(args.bundle) as archive:
            archive.extractall(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["files"].items():
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"hash mismatch: {name}")
        target = root / "site-packages"
        if args.install_wheels:
            wheels = sorted((root / "wheelhouse").glob("*.whl"))
            subprocess.run([
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--target",
                str(target),
                *map(str, wheels),
            ], check=True)
        checkpoint_dir = root / "checkpoint"
        checkpoints = sorted(checkpoint_dir.glob("*.npz"))
        if len(checkpoints) != 1:
            raise RuntimeError(f"expected one numpy checkpoint, found {len(checkpoints)}")
        code = (
            "from drlqn.config import drlqn_xs; "
            "from drlqn.train.submission_solver import SubmissionDRLQNSolver; "
            "from drlqn.train.checkpoint import load_solver_checkpoint; "
            "s=SubmissionDRLQNSolver(drlqn_xs(), seed=0); "
            f"m=load_solver_checkpoint(s, r'{checkpoints[0]}'); "
            "assert m"
        )
        pythonpath = str(root) + (os.pathsep + str(target) if args.install_wheels else "")
        env = {**os.environ, "PYTHONPATH": pythonpath}
        subprocess.run([sys.executable, "-c", code], cwd=root, env=env, check=True)
    print(json.dumps({"bundle": str(args.bundle), "validated": True}, indent=2))


if __name__ == "__main__":
    main()
