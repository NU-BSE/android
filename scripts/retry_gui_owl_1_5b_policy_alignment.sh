#!/usr/bin/env bash
set -euo pipefail

# Retry only the stages that failed to learn/preserve GUI-Owl tool-call syntax.
# Reuses the accepted 16-layer recovery checkpoint; does NOT rerun 28->24->20->16.
# Stops at accepted Hugging Face weights and never performs GGUF conversion.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

DEVICE="${DEVICE:-cuda}"
ON_POLICY_MAX_RECORDS_PER_CORPUS="${ON_POLICY_MAX_RECORDS_PER_CORPUS:-8000}"
ON_POLICY_MAX_UPDATES="${ON_POLICY_MAX_UPDATES:-1500}"
SPECIALIZE_MAX_RECORDS_PER_CORPUS="${SPECIALIZE_MAX_RECORDS_PER_CORPUS:-}"
SPECIALIZE_MAX_UPDATES="${SPECIALIZE_MAX_UPDATES:-1200}"
PACKAGE_WEIGHTS="${PACKAGE_WEIGHTS:-1}"

source_checkpoint="artifacts/students/v2/progressive/1.5b/depth-16-recovered"
on_policy="artifacts/students/v2/progressive/1.5b/on-policy"
final="artifacts/students/v2/gui-owl-1.5b-distilled"
archive_root="artifacts/students/v2/failed-archive/$(date -u +%Y%m%dT%H%M%SZ)"
diagnostics="artifacts/diagnostics/policy-alignment-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$diagnostics"

accepted() {
  local checkpoint="$1"
  python - "$checkpoint/distillation-manifest.json" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
value = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if value.get("accepted") is True else 1)
PY
}

if ! accepted "$source_checkpoint"; then
  echo "error: accepted 16-layer recovery checkpoint is required: $source_checkpoint" >&2
  exit 1
fi

printf '=== GUI-Owl 1.5B policy-alignment retry ===\n'
printf 'source: %s\n' "$source_checkpoint"
printf 'on-policy cap per corpus: %s\n' "$ON_POLICY_MAX_RECORDS_PER_CORPUS"
printf 'on-policy updates: %s\n' "$ON_POLICY_MAX_UPDATES"
printf 'diagnostics: %s\n' "$diagnostics"

if [[ "$DEVICE" == cuda* ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || { echo 'error: nvidia-smi unavailable' >&2; exit 1; }
  nvidia-smi
fi

# Preserve the failed attempt without copying large safetensors. mv is cheap on
# the same filesystem and leaves the accepted depth-16 recovery checkpoint intact.
mkdir -p "$archive_root"
for path in \
  "$on_policy" \
  "${on_policy}-best" \
  "$final" \
  "${final}-best"
do
  if [[ -e "$path" ]]; then
    mv "$path" "$archive_root/$(basename "$path")"
  fi
done
printf 'archived previous failed policy checkpoints under: %s\n' "$archive_root"

echo '==> Re-running on-policy KD from accepted depth-16 recovery'
# The old run mixed the entire 50k broad corpus into a 1,500-update budget.
# Capping each corpus here increases the encounter rate of actual GUI-policy
# records while the config gives GUI examples stronger loss weight. The new
# on-policy gate requires >=50% parseable GUI actions before specialization.
set +e
python scripts/distill_gui_owl_student.py \
  --student 1.5b \
  --stage on_policy \
  --student-model "$source_checkpoint" \
  --teacher-model mPLUG/GUI-Owl-1.5-2B-Instruct \
  --output "$on_policy" \
  --device "$DEVICE" \
  --max-records-per-corpus "$ON_POLICY_MAX_RECORDS_PER_CORPUS" \
  --max-updates "$ON_POLICY_MAX_UPDATES"
on_policy_status=$?
set -e

if [[ -d "$on_policy" ]]; then
  echo '==> Saving concrete on-policy GUI generations'
  python scripts/evaluate_gui_owl_policy_checkpoint.py \
    --checkpoint "$on_policy" \
    --device "$DEVICE" \
    --output "$diagnostics/on-policy-gui.jsonl" || true
fi
if [[ "$on_policy_status" -ne 0 ]]; then
  echo "error: on-policy alignment still failed its generation gate." >&2
  echo "inspect: $diagnostics/on-policy-gui.jsonl" >&2
  exit "$on_policy_status"
fi

if ! accepted "$on_policy"; then
  echo "error: on-policy trainer returned success without an accepted manifest" >&2
  exit 1
fi

echo '==> On-policy checkpoint passed GUI syntax gate; starting specialization with K3 retained'
specialize_args=(
  --student 1.5b
  --stage specialize
  --student-model "$on_policy"
  --teacher-model mPLUG/GUI-Owl-1.5-2B-Instruct
  --output "$final"
  --device "$DEVICE"
  --max-updates "$SPECIALIZE_MAX_UPDATES"
)
if [[ -n "$SPECIALIZE_MAX_RECORDS_PER_CORPUS" ]]; then
  specialize_args+=(--max-records-per-corpus "$SPECIALIZE_MAX_RECORDS_PER_CORPUS")
fi

set +e
python scripts/distill_gui_owl_student.py "${specialize_args[@]}"
specialize_status=$?
set -e

if [[ -d "$final" ]]; then
  echo '==> Saving concrete final GUI generations'
  python scripts/evaluate_gui_owl_policy_checkpoint.py \
    --checkpoint "$final" \
    --device "$DEVICE" \
    --output "$diagnostics/final-gui.jsonl" || true
fi
if [[ "$specialize_status" -ne 0 ]]; then
  echo "error: specialization failed its generation gate." >&2
  echo "inspect: $diagnostics/final-gui.jsonl" >&2
  exit "$specialize_status"
fi

if ! accepted "$final"; then
  echo "error: final 1.5B checkpoint is not accepted" >&2
  exit 1
fi

if [[ "$PACKAGE_WEIGHTS" == "1" ]]; then
  echo '==> Packaging accepted Hugging Face weights only (no GGUF)'
  bash scripts/package_gui_owl_hf_weights.sh 1.5b
fi

printf '\nAccepted 1.5B policy-aligned checkpoint:\n  %s\n' "$final"
printf 'Concrete generation diagnostics:\n  %s\n' "$diagnostics"
printf 'No GGUF conversion was performed.\n'
