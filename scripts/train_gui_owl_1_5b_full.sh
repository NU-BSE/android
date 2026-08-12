#!/usr/bin/env bash
set -euo pipefail

# Full v2 training path for the ~1.5B GUI-Owl student on one CUDA GPU.
# This script stops at the accepted Hugging Face checkpoint and packages only
# those trained weights. It deliberately never invokes GGUF conversion.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-1}"
FORCE_RECOVERY_DATA="${FORCE_RECOVERY_DATA:-0}"
FORCE_CALIBRATION="${FORCE_CALIBRATION:-0}"
PACKAGE_WEIGHTS="${PACKAGE_WEIGHTS:-1}"
RECOVERY_MAX_RECORDS="${RECOVERY_MAX_RECORDS:-50000}"
CALIBRATION_RECORDS="${CALIBRATION_RECORDS:-64}"
DISTILL_INTERMEDIATE_UPDATES="${DISTILL_INTERMEDIATE_UPDATES:-750}"
DISTILL_MAX_RECORDS="${DISTILL_MAX_RECORDS:-}"
DISTILL_MAX_UPDATES="${DISTILL_MAX_UPDATES:-}"

recovery="artifacts/training/gui-owl-recovery.jsonl"
calibration="artifacts/calibration/gui-owl-compression.json"
final="artifacts/students/v2/gui-owl-1.5b-distilled"
log_dir="artifacts/logs/v2"
mkdir -p "$log_dir"
log_file="$log_dir/train-1.5b-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log_file") 2>&1

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

require_file() {
  [[ -f "$1" ]] || { echo "error: missing required file: $1" >&2; exit 1; }
}

printf '=== GUI-Owl v2 full 1.5B training ===\n'
printf 'repo: %s\nlog:  %s\n' "$repo_root" "$log_file"

if [[ "$DEVICE" == cuda* ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || { echo 'error: nvidia-smi is unavailable' >&2; exit 1; }
  nvidia-smi
fi

python - <<'PY'
mods = ["torch", "torchvision", "transformers", "peft", "datasets", "safetensors", "PIL", "android_drlqn"]
missing = []
for name in mods:
    try:
        __import__(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("training environment is incomplete:\n- " + "\n- ".join(missing))
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

# The specialized corpora and trained teacher adapters must already exist.
require_file artifacts/training/gui-policy-unigui-openmobile.jsonl
require_file artifacts/training/chat-screen-qa-qwen.jsonl
require_file artifacts/adapters/gui_policy/adapter_config.json
require_file artifacts/adapters/chat/adapter_config.json

if [[ "$FORCE_RECOVERY_DATA" == "1" || ! -s "$recovery" ]]; then
  echo '==> Preparing broad recovery corpus'
  python scripts/prepare_gui_owl_recovery.py \
    --max-records "$RECOVERY_MAX_RECORDS"
else
  echo "==> Reusing recovery corpus: $recovery"
fi
require_file "$recovery"

if [[ "$FORCE_CALIBRATION" == "1" || ! -s "$calibration" ]]; then
  echo '==> Calibrating 2B teacher for structural pruning'
  python scripts/calibrate_gui_owl_compression.py \
    --device "$DEVICE" \
    --max-records-per-corpus "$CALIBRATION_RECORDS"
else
  echo "==> Reusing calibration manifest: $calibration"
fi
require_file "$calibration"

if [[ "$RESUME" == "1" ]] && accepted "$final"; then
  echo "==> Final 1.5B checkpoint is already accepted; skipping training"
else
  args=(
    --student 1.5b
    --device "$DEVICE"
    --intermediate-recovery-updates "$DISTILL_INTERMEDIATE_UPDATES"
  )
  [[ "$RESUME" == "1" ]] && args+=(--resume)
  [[ -n "$DISTILL_MAX_RECORDS" ]] && args+=(--max-records-per-corpus "$DISTILL_MAX_RECORDS")
  [[ -n "$DISTILL_MAX_UPDATES" ]] && args+=(--max-updates "$DISTILL_MAX_UPDATES")

  echo '==> Running 28 -> 24 -> 20 -> 16 progressive recovery, on-policy KD, specialization'
  python scripts/run_gui_owl_progressive_distillation.py "${args[@]}"
fi

if ! accepted "$final"; then
  echo "error: final 1.5B checkpoint did not pass generation gates: $final" >&2
  exit 1
fi

echo '==> Accepted 1.5B checkpoint'
python - "$final/distillation-manifest.json" <<'PY'
import json
import sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps({
    "accepted": value.get("accepted"),
    "optimizer_updates": value.get("optimizer_updates"),
    "completion_tokens": value.get("completion_tokens"),
    "student_sampled_tokens": value.get("student_sampled_tokens"),
    "final_evaluation": value.get("final_evaluation"),
}, indent=2, sort_keys=True))
PY

if [[ "$PACKAGE_WEIGHTS" == "1" ]]; then
  echo '==> Packaging only final Hugging Face weights; no GGUF conversion'
  bash scripts/package_gui_owl_hf_weights.sh 1.5b
fi

printf '\nTraining complete. Final checkpoint:\n  %s\n' "$final"
printf 'No GGUF files were created by this script.\n'
printf 'Transfer the HF-only bundle from artifacts/transfer/v2-hf/ and convert locally after verification.\n'
