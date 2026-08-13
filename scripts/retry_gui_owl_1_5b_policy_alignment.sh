#!/usr/bin/env bash
set -euo pipefail

# Retry only the stages that failed to learn/preserve GUI-Owl tool-call syntax.
# Reuses the accepted 16-layer recovery checkpoint; does NOT rerun 28->24->20->16.
# Stops at accepted Hugging Face weights and never performs GGUF conversion.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

DEVICE="${DEVICE:-cuda}"
GUI_BRIDGE_MAX_RECORDS_PER_CORPUS="${GUI_BRIDGE_MAX_RECORDS_PER_CORPUS:-4000}"
GUI_BRIDGE_MAX_UPDATES="${GUI_BRIDGE_MAX_UPDATES:-600}"
ON_POLICY_MAX_RECORDS_PER_CORPUS="${ON_POLICY_MAX_RECORDS_PER_CORPUS:-4000}"
ON_POLICY_MAX_UPDATES="${ON_POLICY_MAX_UPDATES:-1500}"
SPECIALIZE_MAX_RECORDS_PER_CORPUS="${SPECIALIZE_MAX_RECORDS_PER_CORPUS:-}"
SPECIALIZE_MAX_UPDATES="${SPECIALIZE_MAX_UPDATES:-1200}"
PACKAGE_WEIGHTS="${PACKAGE_WEIGHTS:-1}"

source_checkpoint="artifacts/students/v2/progressive/1.5b/depth-16-recovered"
bridge="artifacts/students/v2/progressive/1.5b/gui-bridge"
on_policy="artifacts/students/v2/progressive/1.5b/on-policy"
final="artifacts/students/v2/gui-owl-1.5b-distilled"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive_root="artifacts/students/v2/failed-archive/$stamp"
diagnostics="artifacts/diagnostics/policy-alignment-$stamp"
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

check_gui_summary() {
  local summary="$1"
  local label="${2:-GUI}"
  python - "$summary" "$label" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
label = sys.argv[2]
if not path.is_file():
    raise SystemExit(f"missing {label} diagnostic summary: {path}")
x = json.loads(path.read_text(encoding="utf-8"))
failures = []
if float(x.get("first_token_eos_rate", 1.0)) > 0.05:
    failures.append(f"EOS={x.get('first_token_eos_rate')}")
if float(x.get("empty_generation_rate", 1.0)) > 0.05:
    failures.append(f"empty={x.get('empty_generation_rate')}")
if float(x.get("median_generated_tokens", 0.0)) < 8.0:
    failures.append(f"median_tokens={x.get('median_generated_tokens')}")
if float(x.get("gui_parse_rate", 0.0)) < 0.50:
    failures.append(f"gui_parse={x.get('gui_parse_rate')}")
if failures:
    raise SystemExit(label + " gate failed: " + "; ".join(failures))
print(label + " gate passed:", json.dumps(x, sort_keys=True))
PY
}

run_gui_diagnostic() {
  local checkpoint="$1"
  local name="$2"
  local output="$diagnostics/${name}-gui.jsonl"
  python scripts/evaluate_gui_owl_policy_checkpoint.py \
    --checkpoint "$checkpoint" \
    --device "$DEVICE" \
    --output "$output"
  check_gui_summary "${output}.summary.json" "$name"
}

if ! accepted "$source_checkpoint"; then
  echo "error: accepted 16-layer recovery checkpoint is required: $source_checkpoint" >&2
  exit 1
fi

printf '=== GUI-Owl 1.5B policy-alignment retry ===\n'
printf 'source: %s\n' "$source_checkpoint"
printf 'GUI bridge updates: %s\n' "$GUI_BRIDGE_MAX_UPDATES"
printf 'on-policy cap per corpus: %s\n' "$ON_POLICY_MAX_RECORDS_PER_CORPUS"
printf 'on-policy updates: %s\n' "$ON_POLICY_MAX_UPDATES"
printf 'diagnostics: %s\n' "$diagnostics"

if [[ "$DEVICE" == cuda* ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || { echo 'error: nvidia-smi unavailable' >&2; exit 1; }
  nvidia-smi
fi

# PRE-FLIGHT: validate the actual 2B GUI-policy teacher before spending another
# student-training hour. A student cannot recover the Action:/<tool_call> grammar
# through forward/reverse KL when the selected PEFT teacher itself emits EOS or
# malformed GUI outputs. This also verifies the nested adapter path at runtime.
echo '==> Preflight: evaluating 2B GUI-policy teacher adapter'
python scripts/evaluate_gui_owl_policy_teacher.py \
  --device "$DEVICE" \
  --output "$diagnostics/teacher-gui.jsonl"
if ! check_gui_summary "$diagnostics/teacher-gui.jsonl.summary.json" teacher; then
  echo 'error: GUI-policy teacher adapter failed free-running validation.' >&2
  echo 'Do not retrain the compact student yet; inspect/retrain the gui_policy adapter first.' >&2
  echo "inspect: $diagnostics/teacher-gui.jsonl" >&2
  exit 1
fi

# Record the accepted structural checkpoint's GUI behavior as a baseline. It is
# allowed to fail the syntax gate because specialization has not happened yet;
# this file tells us whether the bridge is improving or degrading it.
echo '==> Baseline: evaluating accepted depth-16 recovery checkpoint on GUI prompts'
python scripts/evaluate_gui_owl_policy_checkpoint.py \
  --checkpoint "$source_checkpoint" \
  --device "$DEVICE" \
  --output "$diagnostics/source-depth16-gui.jsonl" || true

# Preserve the failed attempt without copying large safetensors. mv is cheap on
# the same filesystem and leaves the accepted depth-16 recovery checkpoint intact.
mkdir -p "$archive_root"
for path in \
  "$bridge" \
  "${bridge}-best" \
  "$on_policy" \
  "${on_policy}-best" \
  "$final" \
  "${final}-best"
do
  if [[ -e "$path" ]]; then
    mv "$path" "$archive_root/$(basename "$path")"
  fi
done
printf 'archived previous policy checkpoints under: %s\n' "$archive_root"

# The previous bridge used on-policy generation with min_new_tokens=4. That
# forcibly suppressed EOS while sampling, so K3 never observed the exact
# first-token EOS action that the greedy gate was reporting. The config now uses
# min_new_tokens=0: an EOS sample is still a real sampled token and gives K3 a
# gradient against excessive student EOS probability.
echo '==> GUI syntax bridge from accepted depth-16 recovery'
set +e
python scripts/distill_gui_owl_policy_aligned.py \
  --student 1.5b \
  --stage specialize \
  --student-model "$source_checkpoint" \
  --teacher-model mPLUG/GUI-Owl-1.5-2B-Instruct \
  --output "$bridge" \
  --device "$DEVICE" \
  --max-records-per-corpus "$GUI_BRIDGE_MAX_RECORDS_PER_CORPUS" \
  --max-updates "$GUI_BRIDGE_MAX_UPDATES"
bridge_status=$?
set -e

if [[ -d "$bridge" ]]; then
  echo '==> Saving concrete GUI-bridge generations'
  run_gui_diagnostic "$bridge" bridge || bridge_status=1
fi
if [[ "$bridge_status" -ne 0 ]]; then
  echo "error: GUI syntax bridge did not reach the required free-running action grammar." >&2
  echo "inspect: $diagnostics/bridge-gui.jsonl" >&2
  exit "$bridge_status"
fi
if ! accepted "$bridge"; then
  echo "error: GUI bridge returned success without an accepted manifest" >&2
  exit 1
fi

echo '==> Re-running broad on-policy KD from the GUI-aligned bridge'
# Cap each corpus so the 50k broad set cannot crowd GUI-policy records out of a
# finite 1,500-update budget. The policy-aligned entrypoint also stratifies the
# generation holdout so the parse-rate gate always evaluates real GUI examples.
set +e
python scripts/distill_gui_owl_policy_aligned.py \
  --student 1.5b \
  --stage on_policy \
  --student-model "$bridge" \
  --teacher-model mPLUG/GUI-Owl-1.5-2B-Instruct \
  --output "$on_policy" \
  --device "$DEVICE" \
  --max-records-per-corpus "$ON_POLICY_MAX_RECORDS_PER_CORPUS" \
  --max-updates "$ON_POLICY_MAX_UPDATES"
on_policy_status=$?
set -e

if [[ -d "$on_policy" ]]; then
  echo '==> Saving concrete on-policy GUI generations'
  run_gui_diagnostic "$on_policy" on-policy || on_policy_status=1
fi
if [[ "$on_policy_status" -ne 0 ]]; then
  echo "error: on-policy alignment failed to preserve the GUI syntax gate." >&2
  echo "inspect: $diagnostics/on-policy-gui.jsonl" >&2
  exit "$on_policy_status"
fi
if ! accepted "$on_policy"; then
  echo "error: on-policy trainer returned success without an accepted manifest" >&2
  exit 1
fi

echo '==> On-policy checkpoint passed GUI syntax gate; final specialization keeps K3 active'
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
python scripts/distill_gui_owl_policy_aligned.py "${specialize_args[@]}"
specialize_status=$?
set -e

if [[ -d "$final" ]]; then
  echo '==> Saving concrete final GUI generations'
  run_gui_diagnostic "$final" final || specialize_status=1
fi
if [[ "$specialize_status" -ne 0 ]]; then
  echo "error: final specialization failed its GUI/free-generation gate." >&2
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
