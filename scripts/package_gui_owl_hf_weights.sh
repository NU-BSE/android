#!/usr/bin/env bash
set -euo pipefail

# Package only the accepted Hugging Face student checkpoint for transfer off a
# GPU rental. No GGUF conversion is performed and no intermediate checkpoints
# are copied into the transfer bundle.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
student="${1:-${STUDENT_SIZE:-}}"
out_dir="${HF_TRANSFER_OUT_DIR:-$repo_root/artifacts/transfer/v2-hf}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/package_gui_owl_hf_weights.sh <0.5b|1b|1.5b>

Environment:
  HF_TRANSFER_OUT_DIR  output directory (default: artifacts/transfer/v2-hf)

This script packages only the accepted final Hugging Face checkpoint. It does
not call llama.cpp, does not create GGUF, and does not include progressive or
best-checkpoint directories.
EOF
}

case "$student" in
  0.5b|1b|1.5b) ;;
  -h|--help|"") usage; [[ -n "$student" ]] && exit 0 || exit 2 ;;
  *) usage >&2; exit 2 ;;
esac

checkpoint="$repo_root/artifacts/students/v2/gui-owl-${student}-distilled"
manifest="$checkpoint/distillation-manifest.json"
if [[ ! -d "$checkpoint" || ! -f "$manifest" ]]; then
  echo "error: accepted final checkpoint is missing: $checkpoint" >&2
  exit 1
fi

python - "$manifest" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("accepted") is not True:
    raise SystemExit(f"error: checkpoint is not accepted by generation gates: {path}")
PY

if ! compgen -G "$checkpoint/*.safetensors" >/dev/null; then
  echo "error: checkpoint has no safetensors weights: $checkpoint" >&2
  exit 1
fi

mkdir -p "$out_dir"
base="gui-owl-${student}-hf-v2"
bundle="$out_dir/${base}.tar"
checksum="$bundle.sha256"
file_manifest="$out_dir/${base}-files.sha256"

# Safetensors are already dense binary data, so compression usually saves very
# little while wasting paid GPU-host CPU time. Use an uncompressed tar.
rm -f "$bundle" "$checksum" "$file_manifest"

(
  cd "$repo_root"
  tar -cf "$bundle" "artifacts/students/v2/gui-owl-${student}-distilled"
)

# Hash every checkpoint file as well as the transfer tar. These hashes verify
# transfer integrity but do not replace the model's generation acceptance gate.
(
  cd "$checkpoint"
  if command -v sha256sum >/dev/null 2>&1; then
    find . -type f -print0 | sort -z | xargs -0 sha256sum > "$file_manifest"
  else
    find . -type f -print0 | sort -z | xargs -0 shasum -a 256 > "$file_manifest"
  fi
)

(
  cd "$out_dir"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$(basename "$bundle")" > "$(basename "$checksum")"
  else
    shasum -a 256 "$(basename "$bundle")" > "$(basename "$checksum")"
  fi
)

printf 'Accepted HF checkpoint packaged without GGUF:\n'
printf '  %s\n' "$bundle"
printf '  %s\n' "$checksum"
printf '  %s\n' "$file_manifest"
du -h "$bundle" 2>/dev/null || true
