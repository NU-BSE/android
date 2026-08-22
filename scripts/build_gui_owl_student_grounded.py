#!/usr/bin/env python3
"""Build GUI-Owl students with the grounding-preserving depth contract."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_gui_owl_student as base  # noqa: E402
from scripts.gui_owl_grounding_structural import (  # noqa: E402
    protected_text_layer_ids,
    select_salient_layer_ids,
    text_layer_gap_limit,
    text_layer_selection_is_safe,
)

# build_gui_owl_student imported these helpers directly, so patch its module
# globals rather than only changing attributes on android_drlqn.gui_owl_student.
base.protected_text_layer_ids = protected_text_layer_ids
base.select_salient_layer_ids = select_salient_layer_ids
base.text_layer_gap_limit = text_layer_gap_limit
base.text_layer_selection_is_safe = text_layer_selection_is_safe


if __name__ == "__main__":
    raise SystemExit(base.main())
