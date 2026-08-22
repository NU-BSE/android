#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

DEVICE="${DEVICE:-cuda}"
RUN_TAG="${RUN_TAG:-grounded-$(date -u +%Y%m%dT%H%M%SZ)}"
WORK_ROOT="${WORK_ROOT:-artifacts/students/v2/grounding-revision/$RUN_TAG}"
CANDIDATE="${CANDIDATE:-$WORK_ROOT/final}"
REVISION_CONFIG="${REVISION_CONFIG:-$WORK_ROOT/gui-owl-grounding-revision-config.json}"
DISTILL_INTERMEDIATE_UPDATES="${DISTILL_INTERMEDIATE_UPDATES:-1500}"
DISTILL_MAX_RECORDS="${DISTILL_MAX_RECORDS:-}"
RESUME="${RESUME:-0}"
PROMOTE="${PROMOTE:-0}"
CANONICAL="${CANONICAL:-artifacts/students/v2/gui-owl-1.5b-distilled}"

mkdir -p "$WORK_ROOT/logs"
log_file="$WORK_ROOT/logs/retrain-1.5b.log"
exec > >(tee -a "$log_file") 2>&1

require_file() {
  [[ -f "$1" ]] || { echo "error: missing required file: $1" >&2; exit 1; }
}

require_file artifacts/training/gui-owl-recovery.jsonl
require_file artifacts/training/gui-policy-unigui-openmobile.jsonl
require_file artifacts/training/chat-screen-qa-qwen.jsonl
require_file artifacts/calibration/gui-owl-compression.json

export GUI_OWL_DISTILLED_OUTPUT="$CANDIDATE"
export GUI_OWL_REVISION_CONFIG="$REVISION_CONFIG"
export GUI_OWL_FIRST_TOKEN_WEIGHT="${GUI_OWL_FIRST_TOKEN_WEIGHT:-2.0}"

printf '=== GUI-Owl 1.5B grounding revision ===\n'
printf 'run tag:       %s\n' "$RUN_TAG"
printf 'work root:     %s\n' "$WORK_ROOT"
printf 'candidate:     %s\n' "$CANDIDATE"
printf 'revision cfg:  %s\n' "$REVISION_CONFIG"
printf 'first-token w: %s\n' "$GUI_OWL_FIRST_TOKEN_WEIGHT"
printf 'intermediate:  %s updates\n' "$DISTILL_INTERMEDIATE_UPDATES"

if [[ "$DEVICE" == cuda* ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || { echo 'error: nvidia-smi is unavailable' >&2; exit 1; }
  nvidia-smi
fi

args=(
  --student 1.5b
  --device "$DEVICE"
  --work-root "$WORK_ROOT"
  --intermediate-recovery-updates "$DISTILL_INTERMEDIATE_UPDATES"
)
[[ "$RESUME" == "1" ]] && args+=(--resume)
[[ -n "$DISTILL_MAX_RECORDS" ]] && args+=(--max-records-per-corpus "$DISTILL_MAX_RECORDS")

echo '==> Running fresh 28 -> 24 -> 20 grounding-safe compression and distillation'
python scripts/run_gui_owl_progressive_grounded.py "${args[@]}"

[[ -d "$CANDIDATE" ]] || { echo "error: candidate checkpoint missing: $CANDIDATE" >&2; exit 1; }

EVAL_OUT="$WORK_ROOT/final-64.jsonl"
echo '==> Independent 64-record semantic acceptance evaluation'
python scripts/evaluate_gui_owl_policy_checkpoint.py \
  --checkpoint "$CANDIDATE" \
  --device "$DEVICE" \
  --max-records 64 \
  --holdout-fraction 0.05 \
  --seed 0 \
  --output "$EVAL_OUT" \
  --min-action-name-match-rate 0.70 \
  --max-median-coordinate-error 100

python - "$EVAL_OUT.summary.json" <<'PY'
import json
import sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
parse = float(s.get("gui_parse_rate", 0.0))
action = float(s.get("gui_action_name_match_rate", 0.0))
coord = float(s.get("gui_median_coordinate_error", float("inf")))
if parse < 0.70:
    raise SystemExit(f"semantic gate failed: gui_parse_rate={parse:.3f} < 0.70")
if action < 0.70:
    raise SystemExit(f"semantic gate failed: gui_action_name_match_rate={action:.3f} < 0.70")
if coord > 100.0:
    raise SystemExit(f"semantic gate failed: gui_median_coordinate_error={coord:.1f} > 100")
print(json.dumps({
    "gui_parse_rate": parse,
    "gui_action_name_match_rate": action,
    "gui_median_coordinate_error": coord,
    "gui_coordinate_within_50_rate": s.get("gui_coordinate_within_50_rate"),
    "gui_emitted_action_types": s.get("gui_emitted_action_types"),
    "gui_generated_x_min": s.get("gui_generated_x_min"),
    "gui_generated_x_max": s.get("gui_generated_x_max"),
}, indent=2, sort_keys=True))
PY

echo '==> Candidate passed the independent grounding gate'

if [[ "$PROMOTE" == "1" ]]; then
  backup="${CANONICAL}.pre-${RUN_TAG}"
  if [[ -e "$CANONICAL" ]]; then
    [[ ! -e "$backup" ]] || { echo "error: promotion backup already exists: $backup" >&2; exit 1; }
    mv "$CANONICAL" "$backup"
    echo "preserved previous canonical checkpoint: $backup"
  fi
  mv "$CANDIDATE" "$CANONICAL"
  echo "promoted accepted checkpoint: $CANONICAL"
else
  echo "candidate left unpromoted: $CANDIDATE"
  echo "rerun with PROMOTE=1 only if you want automatic promotion after the same gates pass."
fi

printf '\nGrounding revision complete.\nLog: %s\n' "$log_file"
