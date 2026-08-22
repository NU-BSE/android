#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -ne 1 ]]; then
  echo "usage: bash scripts/bootstrap_gui_owl_depth24_contract.sh <grounding-run-root>" >&2
  exit 2
fi

RUN="$1"
DEVICE="${DEVICE:-cuda}"
CHUNK_UPDATES="${CONTRACT_CHUNK_UPDATES:-100}"
MAX_CHUNKS="${CONTRACT_MAX_CHUNKS:-5}"
LEARNING_RATE="${CONTRACT_LEARNING_RATE:-0.000001}"
PREFIX_WEIGHT="${CONTRACT_PREFIX_WEIGHT:-0.25}"
SUFFIX_WEIGHT="${CONTRACT_SUFFIX_WEIGHT:-8.0}"
FOCUS_WEIGHT="${CONTRACT_FOCUS_WEIGHT:-16.0}"
INIT="$RUN/depth-24-init"
BASE_CONFIG="$RUN/gui-owl-grounding-revision-config.json"
BOOT_CONFIG="$RUN/gui-owl-contract-bootstrap-config.json"
BOOT_ROOT="$RUN/depth-24-contract-bootstrap"

[[ -d "$INIT" ]] || { echo "error: missing clean depth-24 init: $INIT" >&2; exit 1; }
[[ -f "$INIT/student-build-manifest.json" ]] || { echo "error: depth-24 init lacks build manifest" >&2; exit 1; }
[[ -f "$BASE_CONFIG" ]] || { echo "error: missing revision config: $BASE_CONFIG" >&2; exit 1; }
[[ "$CHUNK_UPDATES" =~ ^[1-9][0-9]*$ ]] || { echo "error: CONTRACT_CHUNK_UPDATES must be positive integer" >&2; exit 2; }
[[ "$MAX_CHUNKS" =~ ^[1-9][0-9]*$ ]] || { echo "error: CONTRACT_MAX_CHUNKS must be positive integer" >&2; exit 2; }

python - "$BASE_CONFIG" "$BOOT_CONFIG" "$LEARNING_RATE" "$CHUNK_UPDATES" <<'PY'
import json
import sys
from pathlib import Path

src, dst, lr, updates = sys.argv[1:]
value = json.loads(Path(src).read_text(encoding="utf-8"))
stage = value["stages"]["recovery"]
stage.update({
    "supervised_weight": 1.0,
    "forward_kl_weight": 0.25,
    "hidden_state_weight": 0.0,
    "k3_weight": 0.0,
    "learning_rate": float(lr),
    "max_updates": int(updates),
    "gradient_accumulation": 8,
    "freeze_embeddings": True,
    "freeze_lm_head": False,
    "freeze_vision": True,
    "eval_interval_updates": int(updates),
    "eval_records": 32,
    "eval_max_new_tokens": 64,
})
stage["generation_gate"] = {
    "first_token_eos_rate_max": 0.05,
    "empty_generation_rate_max": 0.05,
    "median_generated_tokens_min": 8,
    "gui_parse_rate_min": 0.70,
    "gui_action_name_match_rate_min": 0.70,
    "gui_median_coordinate_error_max": 100.0,
}
value["contract_bootstrap"] = {
    "purpose": "repair Action-to-tool_call autoregressive transition after structural pruning",
    "gui_only": True,
    "lm_head_trainable": True,
    "hidden_state_weight": 0.0,
    "supervised_weight": 1.0,
    "forward_kl_weight": 0.25,
    "chunk_updates": int(updates),
}
Path(dst).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

rm -rf "$BOOT_ROOT"
mkdir -p "$BOOT_ROOT"

source_checkpoint="$INIT"
contract_checkpoint=""
semantic_checkpoint=""

printf '=== GUI-Owl depth-24 tool-call contract bootstrap ===\n'
printf 'run:            %s\n' "$RUN"
printf 'clean init:     %s\n' "$INIT"
printf 'chunk updates:  %s\n' "$CHUNK_UPDATES"
printf 'max chunks:     %s\n' "$MAX_CHUNKS"
printf 'learning rate:  %s\n' "$LEARNING_RATE"
printf 'token weights:  prefix=%s suffix=%s focus=%s\n' "$PREFIX_WEIGHT" "$SUFFIX_WEIGHT" "$FOCUS_WEIGHT"
printf 'LM head:        trainable\n'

for ((chunk=1; chunk<=MAX_CHUNKS; chunk++)); do
  output="$BOOT_ROOT/chunk-$chunk"
  echo
  echo "==> contract chunk $chunk/$MAX_CHUNKS from $source_checkpoint"

  GUI_OWL_CONTRACT_PREFIX_WEIGHT="$PREFIX_WEIGHT" \
  GUI_OWL_CONTRACT_SUFFIX_WEIGHT="$SUFFIX_WEIGHT" \
  GUI_OWL_CONTRACT_FOCUS_WEIGHT="$FOCUS_WEIGHT" \
  python scripts/distill_gui_owl_contract_bootstrap.py \
    --student 1.5b \
    --stage recovery \
    --student-model "$source_checkpoint" \
    --teacher-model mPLUG/GUI-Owl-1.5-2B-Instruct \
    --output "$output" \
    --device "$DEVICE" \
    --max-updates "$CHUNK_UPDATES" \
    --learning-rate "$LEARNING_RATE" \
    --config "$BOOT_CONFIG" \
    --allow-gate-failure

  python - "$output/distillation-manifest.json" <<'PY'
import json
import sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
x = m.get("final_evaluation") or {}
print(json.dumps({
    "semantic_accepted": m.get("accepted"),
    "update": m.get("optimizer_updates"),
    "ce": x.get("mean_supervised_ce"),
    "kl": x.get("mean_teacher_kl"),
    "action_prefix": x.get("gui_action_prefix_rate"),
    "tool_open": x.get("gui_tool_call_open_rate"),
    "tool_close": x.get("gui_tool_call_close_rate"),
    "parse": x.get("gui_parse_rate"),
    "action_match": x.get("gui_action_name_match_rate"),
    "coordinate_median": x.get("gui_median_coordinate_error"),
    "actions": x.get("gui_emitted_action_types"),
    "x_min": x.get("gui_generated_x_min"),
    "x_max": x.get("gui_generated_x_max"),
    "gate_failures": m.get("final_gate_failures"),
}, indent=2, sort_keys=True))
PY

  read -r semantic_ready contract_ready no_progress <<<"$(python - "$output/distillation-manifest.json" "$chunk" <<'PY'
import json
import sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
chunk = int(sys.argv[2])
x = m.get("final_evaluation") or {}
open_rate = float(x.get("gui_tool_call_open_rate") or 0.0)
close_rate = float(x.get("gui_tool_call_close_rate") or 0.0)
parse = float(x.get("gui_parse_rate") or 0.0)
semantic = m.get("accepted") is True
contract = open_rate >= 0.75 and close_rate >= 0.75 and parse >= 0.50
no_progress = chunk >= 2 and open_rate == 0.0 and close_rate == 0.0 and parse == 0.0
print(int(semantic), int(contract), int(no_progress))
PY
)"

  if [[ "$semantic_ready" == "1" ]]; then
    semantic_checkpoint="$output"
    contract_checkpoint="$output"
    break
  fi
  if [[ "$contract_ready" == "1" ]]; then
    contract_checkpoint="$output"
    break
  fi
  if [[ "$no_progress" == "1" ]]; then
    echo >&2
    echo "error: no tool-call transition appeared after $chunk chunks; stopping early." >&2
    echo "Do not continue this objective without inspecting the token/rank diagnostic." >&2
    exit 1
  fi

  source_checkpoint="$output"
done

if [[ -z "$contract_checkpoint" ]]; then
  echo >&2
  echo "error: contract bootstrap did not reach tool_open/tool_close >= 75% and parse >= 50%." >&2
  echo "Checkpoints remain under: $BOOT_ROOT" >&2
  exit 1
fi

echo
echo "contract bootstrap checkpoint: $contract_checkpoint"
if [[ -n "$semantic_checkpoint" ]]; then
  echo "It also passed the full action/coordinate semantic gate."
else
  echo "The tool-call contract recovered, but the full semantic gate has not yet passed."
  echo "Do not resume 24->20 yet; use this checkpoint for the next grounding-only stage."
fi
