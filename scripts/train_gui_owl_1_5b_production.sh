#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

DEVICE="${DEVICE:-cuda}"
RUN_TAG="${RUN_TAG:-prod-$(date -u +%Y%m%dT%H%M%SZ)}"
WORK_ROOT="${WORK_ROOT:-artifacts/students/v2/production-1.5b/$RUN_TAG}"
PROMOTE="${PROMOTE:-1}"
RESUME="${RESUME:-0}"
CONTRACT_CHUNK_UPDATES="${CONTRACT_CHUNK_UPDATES:-100}"
CONTRACT_MAX_CHUNKS="${CONTRACT_MAX_CHUNKS:-6}"
DEPTH24_RECOVERY_UPDATES="${DEPTH24_RECOVERY_UPDATES:-1800}"
DEPTH20_RECOVERY_UPDATES="${DEPTH20_RECOVERY_UPDATES:-2800}"
ON_POLICY_UPDATES="${ON_POLICY_UPDATES:-1000}"
SPECIALIZE_UPDATES="${SPECIALIZE_UPDATES:-900}"
MAX_RECORDS_PER_CORPUS="${MAX_RECORDS_PER_CORPUS:-}"

if [[ "$PROMOTE" != "0" && "$PROMOTE" != "1" ]]; then
  echo "error: PROMOTE must be 0 or 1" >&2
  exit 2
fi
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
  echo "error: RESUME must be 0 or 1" >&2
  exit 2
fi

printf '=== GUI-Owl 1.5B production training ===\n'
printf 'git head:     %s\n' "$(git rev-parse HEAD)"
printf 'device:       %s\n' "$DEVICE"
printf 'run tag:      %s\n' "$RUN_TAG"
printf 'work root:    %s\n' "$WORK_ROOT"
printf 'promote:      %s\n' "$PROMOTE"
printf 'resume:       %s\n' "$RESUME"
printf 'contract:     %s updates x max %s chunks\n' "$CONTRACT_CHUNK_UPDATES" "$CONTRACT_MAX_CHUNKS"
printf 'depth24 rec:  %s updates\n' "$DEPTH24_RECOVERY_UPDATES"
printf 'depth20 rec:  %s updates\n' "$DEPTH20_RECOVERY_UPDATES"
printf 'on-policy:    %s updates\n' "$ON_POLICY_UPDATES"
printf 'specialize:   %s updates\n' "$SPECIALIZE_UPDATES"

# Fail before loading a teacher or allocating GPU memory if the new production
# entrypoints contain a syntax/import-path mistake.
python -m py_compile \
  android_drlqn/gui_owl_contract.py \
  scripts/gui_owl_contract_loss.py \
  scripts/distill_gui_owl_production.py \
  scripts/evaluate_gui_owl_policy_production.py \
  scripts/gui_owl_grounded_evaluation.py \
  scripts/gui_owl_grounding_structural.py \
  scripts/build_gui_owl_student_grounded.py \
  scripts/run_gui_owl_1_5b_production.py

if [[ "$DEVICE" == cuda* ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "error: CUDA requested but nvidia-smi is unavailable" >&2
    exit 1
  }
  nvidia-smi
fi

args=(
  --device "$DEVICE"
  --run-tag "$RUN_TAG"
  --work-root "$WORK_ROOT"
  --contract-chunk-updates "$CONTRACT_CHUNK_UPDATES"
  --contract-max-chunks "$CONTRACT_MAX_CHUNKS"
  --depth24-recovery-updates "$DEPTH24_RECOVERY_UPDATES"
  --depth20-recovery-updates "$DEPTH20_RECOVERY_UPDATES"
  --on-policy-updates "$ON_POLICY_UPDATES"
  --specialize-updates "$SPECIALIZE_UPDATES"
)
[[ "$PROMOTE" == "1" ]] && args+=(--promote)
[[ "$RESUME" == "1" ]] && args+=(--resume)
[[ -n "$MAX_RECORDS_PER_CORPUS" ]] && args+=(--max-records-per-corpus "$MAX_RECORDS_PER_CORPUS")

exec python scripts/run_gui_owl_1_5b_production.py "${args[@]}"
