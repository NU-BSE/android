#!/usr/bin/env bash
set -euo pipefail

# Package accepted v2 students for transfer off a disposable GPU host.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
student_selector="${1:-${STUDENT_SIZE:-all}}"
student_config="${STUDENT_CONFIG:-$repo_root/config/gui_owl_students.json}"
gguf_dir="${GGUF_OUT_DIR:-$repo_root/artifacts/gguf/v2}"
transfer_dir="${TRANSFER_OUT_DIR:-$repo_root/artifacts/transfer/v2}"
quant="${TRANSFER_GGUF_QUANT:-Q4_0}"
outtype="${GGUF_OUTTYPE:-bf16}"
include_hf="${INCLUDE_HF_CHECKPOINT:-1}"
include_full_gguf="${INCLUDE_FULL_GGUF:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/package_gui_owl_students.sh [0.5b|1b|1.5b|all]

Environment:
  STUDENT_CONFIG            student config JSON
  GGUF_OUT_DIR              default: artifacts/gguf/v2
  TRANSFER_OUT_DIR          default: artifacts/transfer/v2
  TRANSFER_GGUF_QUANT       default: Q4_0
  GGUF_OUTTYPE              mmproj/full precision (default: bf16)
  INCLUDE_HF_CHECKPOINT=0   omit the full HF source checkpoint
  INCLUDE_FULL_GGUF=1       also include the unquantized language GGUF

Each size gets an uncompressed .tar plus .tar.sha256. The exporter must already
have accepted and converted the v2 checkpoint.
EOF
}

case "$student_selector" in
  0.5b|1b|1.5b) sizes=("$student_selector") ;;
  all) sizes=(0.5b 1b 1.5b) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

resolve_source_rel() {
  local size="$1"
  python - "$student_config" "$size" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
path = config.get("students", {}).get(sys.argv[2], {}).get("distilled")
if not path:
    raise SystemExit(f"student config lacks distilled path for {sys.argv[2]}")
print(path)
PY
}

mkdir -p "$transfer_dir"
quant_lower="$(printf '%s' "$quant" | tr '[:upper:]' '[:lower:]')"

for size in "${sizes[@]}"; do
  base="gui-owl-${size}"
  hf_rel="$(resolve_source_rel "$size")"
  if [[ "$hf_rel" = /* ]]; then
    echo "error: packaging expects repository-relative student paths: $hf_rel" >&2
    exit 1
  fi
  runtime_rel="${gguf_dir#$repo_root/}/${base}-${quant_lower}.gguf"
  mmproj_rel="${gguf_dir#$repo_root/}/mmproj-${base}-${outtype}.gguf"
  manifest_rel="${gguf_dir#$repo_root/}/${base}-export-manifest.json"
  full_rel="${gguf_dir#$repo_root/}/${base}-${outtype}.gguf"

  for rel in "$runtime_rel" "$mmproj_rel" "$manifest_rel"; do
    if [[ ! -f "$repo_root/$rel" ]]; then
      echo "error: required $size artifact is missing: $rel" >&2
      echo "run: bash scripts/export_gui_owl_gguf.sh $size" >&2
      exit 1
    fi
  done
  if [[ "$include_hf" == "1" && ! -d "$repo_root/$hf_rel" ]]; then
    echo "error: accepted distilled HF checkpoint is missing: $hf_rel" >&2
    exit 1
  fi
  if [[ "$include_full_gguf" == "1" && ! -f "$repo_root/$full_rel" ]]; then
    echo "error: full GGUF requested but missing: $full_rel" >&2
    exit 1
  fi

  files=("$runtime_rel" "$mmproj_rel" "$manifest_rel" "config/gui_owl_students.json")
  if [[ "$include_hf" == "1" ]]; then
    files+=("$hf_rel")
  fi
  if [[ "$include_full_gguf" == "1" ]]; then
    files+=("$full_rel")
  fi

  bundle="$transfer_dir/${base}-mac-bundle.tar"
  checksum="$bundle.sha256"
  echo "==> [$size] Creating $(basename "$bundle")"
  rm -f "$bundle" "$checksum"
  tar -C "$repo_root" -cf "$bundle" "${files[@]}"
  (
    cd "$transfer_dir"
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum "$(basename "$bundle")" > "$(basename "$checksum")"
    else
      shasum -a 256 "$(basename "$bundle")" > "$(basename "$checksum")"
    fi
  )
  du -h "$bundle"
done

printf '\nTransfer bundles are ready in %s\n' "$transfer_dir"
printf 'Do not destroy the GPU instance until local SHA-256 verification and inference testing pass.\n'
