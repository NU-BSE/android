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
GUI_BRIDGE_CANARY_UPDATES="${GUI_BRIDGE_CANARY_UPDATES:-100}"
GUI_BRIDGE_MAX_UPDATES="${GUI_BRIDGE_MAX_UPDATES:-500}"
ON_POLICY_MAX_RECORDS_PER_CORPUS="${ON_POLICY_MAX_RECORDS_PER_CORPUS:-4000}"
ON_POLICY_MAX_UPDATES="${ON_POLICY_MAX_UPDATES:-1500}"
SPECIALIZE_MAX_RECORDS_PER_CORPUS="${SPECIALIZE_MAX_RECORDS_PER_CORPUS:-}"
SPECIALIZE_MAX_UPDATES="${SPECIALIZE_MAX_UPDATES:-1200}"
PACKAGE_WEIGHTS="${PACKAGE_WEIGHTS:-1}"

source_checkpoint="artifacts/students/v2/progressive/1.5b/depth-16-recovered"
canary="artifacts/students/v2/progressive/1.5b/gui-bridge-canary"
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

check_canary_progress() {
  local before="$1"
  local after="$2"
  python - "$before" "$after" <<'PY'
import json
import sys
from pathlib import Path
before = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
after = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

def f(x, name, default):
    return float(x.get(name, default))

before_eos = f(before, "first_token_eos_rate", 1.0)
after_eos = f(after, "first_token_eos_rate", 1.0)
before_empty = f(before, "empty_generation_rate", 1.0)
after_empty = f(after, "empty_generation_rate", 1.0)
before_parse = f(before, "gui_parse_rate", 0.0)
after_parse = f(after, "gui_parse_rate", 0.0)
before_median = f(before, "median_generated_tokens", 0.0)
after_median = f(after, "median_generated_tokens", 0.0)

improved = (
    after_eos < before_eos
    or after_empty < before_empty
    or after_parse > before_parse
    or after_median > before_median
)
print("canary before:", json.dumps(before, sort_keys=True))
print("canary after: ", json.dumps(after, sort_keys=True))
if not improved:
    raise SystemExit(
        "GUI bridge canary made no free-generation progress; refusing another 500-update paid run"
    )
print("GUI bridge canary improved at least one free-generation metric; continuing")
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
}

if ! accepted "$source_checkpoint"; then
  echo "error: accepted 16-layer recovery checkpoint is required: $source_checkpoint" >&2
  exit 1
fi

printf '=== GUI-Owl 1.5B policy-alignment retry ===\n'
printf 'source: %s\n' "$source_checkpoint"
printf 'GUI bridge canary updates: %s\n' "$GUI_BRIDGE_CANARY_UPDATES"
printf 'GUI bridge continuation updates: %s\n' "$GUI_BRIDGE_MAX_UPDATES"
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
# the canary must improve at least one free-generation metric over this baseline.
echo '==> Baseline: evaluating accepted depth-16 recovery checkpoint on GUI prompts'
run_gui_diagnostic "$source_checkpoint" source-depth16

# Preserve failed policy attempts without copying large safetensors. mv is cheap
# on the same filesystem and leaves the accepted depth-16 recovery checkpoint intact.
mkdir -p "$archive_root"
for path in \
  "$canary" \
  "${canary}-best" \
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

# Cheap canary before another full bridge. The policy-aligned entrypoint fixes
# the old zero-gradient K3 clamp, upweights the first completion token, and this
# bridge trains on GUI policy records only. --allow-gate-failure lets us inspect
# movement after 100 updates instead of requiring the final 50%% parse gate yet.
echo '==> Canary: 100-update GUI-only syntax recovery'
GUI_OWL_ONLY_GUI=1 python scripts/distill_gui_owl_policy_aligned.py \
  --student 1.5b \
  --stage specialize \
  --student-model "$source_checkpoint" \
  --teacher-model mPLUG/GUI-Owl-1.5-2B-Instruct \
  --output "$canary" \
  --device "$DEVICE" \
  --max-records-per-corpus "$GUI_BRIDGE_MAX_RECORDS_PER_CORPUS" \
  --max-updates "$GUI_BRIDGE_CANARY_UPDATES" \
  --allow-gate-failure

echo '==> Saving concrete canary generations'
run_gui_diagnostic "$canary" canary
check_canary_progress \
  "$diagnostics/source-depth16-gui.jsonl.summary.json" \
  "$diagnostics/canary-gui.jsonl.summary.json"

# Continue from the canary rather than paying for those first 100 updates twice.
echo '==> GUI-only syntax bridge continuation'
set +e
GUI_OWL_ONLY_GUI=1 python scripts/distill_gui_owl_policy_aligned.py \
  --student 1.5b \
  --stage specialize \
  --student-model "$canary" \
  --teacher-model mPLUG/GUI-Owl-1.5-2B-Instruct \
  --output "$bridge" \
  --device "$DEVICE" \
  --max-records-per-corpus "$GUI_BRIDGE_MAX_RECORDS_PER_CORPUS" \
  --max-updates "$GUI_BRIDGE_MAX_UPDATES"
bridge_status=$?
set -e

if [[ -d "$bridge" ]]; then
  echo '==> Saving concrete GUI-bridge generations'
  run_gui_diagnostic "$bridge" bridge
  check_gui_summary "$diagnostics/bridge-gui.jsonl.summary.json" bridge || bridge_status=1
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
  run_gui_diagnostic "$on_policy" on-policy
  check_gui_summary "$diagnostics/on-policy-gui.jsonl.summary.json" on-policy || on_policy_status=1
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
  run_gui_diagnostic "$final" final
  check_gui_summary "$diagnostics/final-gui.jsonl.summary.json" final || specialize_status=1
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
