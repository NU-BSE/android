#!/usr/bin/env bash
set -euo pipefail

# Package each distilled student for transfer off a disposable GPU host.
# By default the bundle contains the source Hugging Face checkpoint plus the
# mobile Q4_0 GGUF, BF16 mmproj, and export manifest. The large BF16 language
# GGUF is excluded because it is reproducible from the HF checkpoint.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
student_selector="${1:-${STUDENT_SIZE:-all}}"
gguf_dir="${GGUF_OUT_DIR:-$repo_root/artifacts/gguf}"
transfer_dir="${TRANSFER_OUT_DIR:-$repo_root/artifacts/transfer}"
quant="${TRANSFER_GGUF_QUANT:-Q4_0}"
outtype="${GGUF_OUTTYPE:-bf16}"
include_hf="${INCLUDE_HF_CHECKPOINT:-1}"
include_full_gguf="${INCLUDE_FULL_GGUF:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/package_gui_owl_students.sh [0.5b|1b|1.5b|all]

Environment:
  TRANSFER_OUT_DIR         bundles directory (default: artifacts/transfer)
  TRANSFER_GGUF_QUANT      runtime GGUF quantization (default: Q4_0)
  GGUF_OUTTYPE             mmproj/full conversion precision (default: bf16)
  INCLUDE_HF_CHECKPOINT=0  omit the full HF source checkpoint
  INCLUDE_FULL_GGUF=1      also include the large unquantized language GGUF

Each size gets its own uncompressed .tar plus .tar.sha256. GGUF files are
already dense binary data, so uncompressed tar avoids wasting GPU rental time
on low-yield recompression and works with macOS tar directly.
EOF
}

case "$student_selector" in
  0.5b|1b|1.5b) sizes=("$student_selector") ;;
  all) sizes=(0.5b 1b 1.5b) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

mkdir -p "$transfer_dir"
quant_lower="$(printf '%s' "$quant" | tr '[:upper:]' '[:lower:]')"

for size in "${sizes[@]}"; do
  base="gui-owl-${size}"
  hf_rel="artifacts/students/${base}-distilled"
  runtime_rel="artifacts/gguf/${base}-${quant_lower}.gguf"
  mmproj_rel="artifacts/gguf/mmproj-${base}-${outtype}.gguf"
  manifest_rel="artifacts/gguf/${base}-export-manifest.json"
  full_rel="artifacts/gguf/${base}-${outtype}.gguf"

  required=("$runtime_rel" "$mmproj_rel" "$manifest_rel")
  for rel in "${required[@]}"; do
    if [[ ! -f "$repo_root/$rel" ]]; then
      echo "error: required $size artifact is missing: $rel" >&2
      echo "run: bash scripts/export_gui_owl_gguf.sh $size" >&2
      exit 1
    fi
  done
  if [[ "$include_hf" == "1" && ! -d "$repo_root/$hf_rel" ]]; then
    echo "error: distilled HF checkpoint is missing: $hf_rel" >&2
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

echo
printf 'Transfer bundles are ready in %s\n' "$transfer_dir"
printf 'Do not destroy the Vast instance until the Mac-side script verifies every SHA-256.\n'
