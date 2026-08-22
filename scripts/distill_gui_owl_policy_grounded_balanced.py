#!/usr/bin/env python3
"""Policy-aligned GUI-Owl distillation with GUI action balancing."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importing the policy-aligned entrypoint installs the grounded evaluator,
# stratified GUI evaluation, live-gradient K3 clipping, and first-token repair.
from scripts import distill_gui_owl_policy_aligned as policy  # noqa: E402
from scripts.gui_owl_grounding_revision import balanced_corpus_records  # noqa: E402

_ORIGINAL_CORPUS_RECORDS = policy.base._corpus_records


def _balanced_records(sources, max_records):
    return balanced_corpus_records(_ORIGINAL_CORPUS_RECORDS, sources, max_records)


policy.base._corpus_records = _balanced_records


if __name__ == "__main__":
    raise SystemExit(policy.base.main())
