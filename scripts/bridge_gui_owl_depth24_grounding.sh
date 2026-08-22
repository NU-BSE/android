#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -ne 1 ]]; then
  echo "usage: bash scripts/bridge_gui_owl_depth24_grounding.sh <grounding-run-root>" >&2
  exit 2
fi

RUN="$1"
DEVICE="${DEVICE:-cuda}"
CHUNK_UPDATES="${BRIDGE_CHUNK_UPDATES:-250}"
MAX_CHUNKS="${BRIDGE_MAX_CHUNKS:-4}"
FIRST_TOKEN_WEIGHT="${BRIDGE_FIRST_TOKEN_WEIGHT:-6.0}"
LEARNING_RATE="${BRIDGE_LEARNING_RATE:-0.000001}"
INIT="$RUN/depth-24-init"
FAILED_RECOVERY="$RUN/depth-24-recovered"
BASE_CONFIG="$RUN/gui-owl-grounding-revision-config.json"
BRIDGE_CONFIG="$RUN/gui-owl-grounding-bridge-config.json"
BRIDGE_ROOT="$RUN/depth-24-gui-bridge"

[[ -d "$INIT" ]] || { echo "error: missing depth-24 init: $INIT" >&2; exit 1; }
[[ -f "$INIT/student-build-manifest.json" ]] || { echo "error: depth-24 init lacks build manifest" >&2; exit 1; }
[[ -f "$BASE_CONFIG" ]] || { echo "error: missing revision config: $BASE_CONFIG" >&2; exit 1; }
[[ "$CHUNK_UPDATES" =~ ^[1-9][0-9]*$ ]] || { echo "error: BRIDGE_CHUNK_UPDATES must be positive integer" >&2; exit 2; }
[[ "$MAX_CHUNKS" =~ ^[1-9][0-9]*$ ]] || { echo "error: BRIDGE_MAX_CHUNKS must be positive integer" >&2; exit 2; }

python - "$BASE_CONFIG" "$BRIDGE_CONFIG" "$LEARNING_RATE" "$CHUNK_UPDATES" <<'PY'
import json
import sys
from pathlib import Path

src, dst, lr, chunk_updates = sys.argv[1:]
value = json.loads(Path(src).read_text(encoding="utf-8"))
stage = value["stages"]["recovery"]
stage.update({
    "supervised_weight": 0.75,
    "forward_kl_weight": 1.0,
    "hidden_state_weight": 0.10,
    "k3_weight": 0.0,
    "learning_rate": float(lr),
    "max_updates": int(chunk_updates),
    "eval_interval_updates": max(1, int(chunk_updates) // 2),
    "eval_records": 64,
})
stage["generation_gate"] = {
    "first_token_eos_rate_max": 0.05,
    "empty_generation_rate_max": 0.05,
    "median_generated_tokens_min": 8,
    "gui_parse_rate_min": 0.70,
    "gui_action_name_match_rate_min": 0.70,
    "gui_median_coordinate_error_max": 100.0,
}
value["grounding_bridge"] = {
    "gui_only": True,
    "first_token_weight": 6.0,
    "chunk_updates": int(chunk_updates),
    "max_chunks": None,
    "purpose": "restore free-running GUI contract before mixed structural recovery",
}
Path(dst).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

mkdir -p "$BRIDGE_ROOT"

source_checkpoint="$INIT"
passed_checkpoint=""

printf '=== GUI-Owl depth-24 GUI-only grounding bridge ===\n'
printf 'run:                %s\n' "$RUN"
printf 'initial checkpoint: %s\n' "$INIT"
printf 'chunk updates:      %s\n' "$CHUNK_UPDATES"
printf 'max chunks:         %s\n' "$MAX_CHUNKS"
printf 'first-token weight: %s\n' "$FIRST_TOKEN_WEIGHT"
printf 'learning rate:      %s\n' "$LEARNING_RATE"

for ((chunk=1; chunk<=MAX_CHUNKS; chunk++)); do
  output="$BRIDGE_ROOT/chunk-$chunk"
  echo
  echo "==> bridge chunk $chunk/$MAX_CHUNKS from $source_checkpoint"

  GUI_OWL_ONLY_GUI=1 \
  GUI_OWL_FIRST_TOKEN_WEIGHT="$FIRST_TOKEN_WEIGHT" \
  python scripts/distill_gui_owl_policy_grounded_balanced.py \
    --student 1.5b \
    --stage recovery \
    --student-model "$source_checkpoint" \
    --teacher-model mPLUG/GUI-Owl-1.5-2B-Instruct \
    --output "$output" \
    --device "$DEVICE" \
    --max-updates "$CHUNK_UPDATES" \
    --learning-rate "$LEARNING_RATE" \
    --config "$BRIDGE_CONFIG" \
    --allow-gate-failure

  python - "$output/distillation-manifest.json" <<'PY'
import json
import sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
x = m.get("final_evaluation") or {}
print(json.dumps({
    "accepted": m.get("accepted"),
    "update": m.get("optimizer_updates"),
    "parse": x.get("gui_parse_rate"),
    "action_match": x.get("gui_action_name_match_rate"),
    "coordinate_median": x.get("gui_median_coordinate_error"),
    "within_50": x.get("gui_coordinate_within_50_rate"),
    "actions": x.get("gui_emitted_action_types"),
    "x_min": x.get("gui_generated_x_min"),
    "x_max": x.get("gui_generated_x_max"),
    "gate_failures": m.get("final_gate_failures"),
}, indent=2, sort_keys=True))
PY

  if python - "$output/distillation-manifest.json" <<'PY'
import json
import sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if m.get("accepted") is True else 1)
PY
  then
    passed_checkpoint="$output"
    break
  fi

  source_checkpoint="$output"
done

if [[ -z "$passed_checkpoint" ]]; then
  echo >&2
  echo "error: GUI-only bridge did not cross the semantic gate after $MAX_CHUNKS chunks." >&2
  echo "The original depth-24 init and failed mixed recovery were left untouched." >&2
  exit 1
fi

backup="$RUN/depth-24-recovered.failed-mixed-before-bridge"
if [[ -e "$FAILED_RECOVERY" ]]; then
  if [[ -e "$backup" ]]; then
    echo "error: recovery backup already exists: $backup" >&2
    exit 1
  fi
  mv "$FAILED_RECOVERY" "$backup"
  echo "preserved failed mixed recovery: $backup"
fi

mv "$passed_checkpoint" "$FAILED_RECOVERY"
echo "installed accepted GUI bridge as resume checkpoint: $FAILED_RECOVERY"
echo
echo "Next command:"
echo "  WORK_ROOT='$RUN' RESUME=1 PROMOTE=1 bash scripts/retrain_gui_owl_1_5b_grounded.sh"
