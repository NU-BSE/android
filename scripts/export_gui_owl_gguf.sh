#!/usr/bin/env bash
set -euo pipefail

# Convert one or all distilled GUI-Owl students to llama.cpp GGUF plus mmproj.
# Intended for the Vast.ai training host after distillation finishes.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
student_selector="${1:-${STUDENT_SIZE:-all}}"
llama_cpp_dir="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
llama_cpp_ref="${LLAMA_CPP_REF:-b9637}"
out_dir="${GGUF_OUT_DIR:-$repo_root/artifacts/gguf}"
outtype="${GGUF_OUTTYPE:-bf16}"
quants="${GGUF_QUANTS:-Q4_0}"
force="${GGUF_FORCE:-0}"
install_deps="${LLAMA_CPP_INSTALL_DEPS:-0}"
build_quantizer="${LLAMA_CPP_BUILD_QUANTIZER:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/export_gui_owl_gguf.sh [0.5b|1b|1.5b|all]

Environment:
  LLAMA_CPP_DIR              llama.cpp checkout (default: ~/llama.cpp)
  LLAMA_CPP_REF              ref cloned when checkout is absent (default: b9637)
  LLAMA_CPP_INSTALL_DEPS=1   allow pip install -r llama.cpp/requirements.txt
  LLAMA_CPP_BUILD_QUANTIZER  build llama-quantize when missing (default: 1)
  GGUF_OUT_DIR               output directory (default: artifacts/gguf)
  GGUF_OUTTYPE               conversion precision (default: bf16)
  GGUF_QUANTS                space-separated quantizations (default: Q4_0)
  GGUF_FORCE=1               overwrite existing GGUF files

The multimodal projector stays at GGUF_OUTTYPE precision. Only the language
model is quantized by this script.
EOF
}

case "$student_selector" in
  0.5b|1b|1.5b) sizes=("$student_selector") ;;
  all) sizes=(0.5b 1b 1.5b) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

for command in git python cmake; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "error: required command not found: $command" >&2
    exit 1
  fi
done

if [[ ! -d "$llama_cpp_dir/.git" ]]; then
  echo "==> Cloning llama.cpp at $llama_cpp_ref into $llama_cpp_dir"
  mkdir -p "$(dirname "$llama_cpp_dir")"
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$llama_cpp_dir"
  git -C "$llama_cpp_dir" fetch --depth 1 origin "$llama_cpp_ref"
  git -C "$llama_cpp_dir" checkout --detach FETCH_HEAD
fi

converter="$llama_cpp_dir/convert_hf_to_gguf.py"
if [[ ! -f "$converter" ]]; then
  echo "error: llama.cpp converter not found: $converter" >&2
  exit 1
fi
if ! grep -q -- '--mmproj' "$converter"; then
  echo "error: this llama.cpp checkout does not expose --mmproj" >&2
  exit 1
fi

if ! python "$converter" --help >/dev/null 2>&1; then
  if [[ "$install_deps" == "1" ]]; then
    echo "==> Installing llama.cpp Python conversion requirements"
    python -m pip install -r "$llama_cpp_dir/requirements.txt"
  else
    cat >&2 <<EOF
error: llama.cpp conversion dependencies are missing.
Run:
  python -m pip install -r '$llama_cpp_dir/requirements.txt'
or rerun with LLAMA_CPP_INSTALL_DEPS=1.
EOF
    exit 1
  fi
fi

quantizer="$llama_cpp_dir/build/bin/llama-quantize"
if [[ ! -x "$quantizer" ]]; then
  if [[ "$build_quantizer" != "1" ]]; then
    echo "error: llama-quantize is missing: $quantizer" >&2
    exit 1
  fi
  echo "==> Building CPU llama-quantize"
  cmake -S "$llama_cpp_dir" -B "$llama_cpp_dir/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF \
    -DLLAMA_CURL=OFF
  cmake --build "$llama_cpp_dir/build" --target llama-quantize \
    --parallel "${LLAMA_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
fi

mkdir -p "$out_dir"
llama_commit="$(git -C "$llama_cpp_dir" rev-parse HEAD)"

do_convert() {
  local size="$1"
  local source="$repo_root/artifacts/students/gui-owl-${size}-distilled"
  local base="gui-owl-${size}"
  local full="$out_dir/${base}-${outtype}.gguf"
  local mmproj="$out_dir/mmproj-${base}-${outtype}.gguf"
  local rebuild_full=0

  if [[ ! -f "$source/config.json" ]]; then
    echo "error: distilled $size checkpoint is missing: $source/config.json" >&2
    return 1
  fi
  if ! compgen -G "$source/*.safetensors" >/dev/null; then
    echo "error: distilled $size checkpoint has no safetensors weights: $source" >&2
    return 1
  fi
  if [[ "$full" == "$mmproj" ]]; then
    echo "error: model and mmproj output paths unexpectedly collide" >&2
    return 1
  fi

  if [[ "$force" == "1" || ! -s "$full" ]]; then
    rebuild_full=1
  elif [[ ! -s "$mmproj" ]]; then
    # Older versions of this exporter passed --outfile "$full" to the mmproj
    # conversion. llama.cpp b9637 does not add a prefix for an explicit custom
    # outfile, so that run could overwrite the language GGUF with the mmproj.
    # Rebuild the language GGUF whenever we see that legacy failure pattern.
    echo "==> [$size] mmproj is missing; rebuilding the language GGUF to repair a possible legacy overwrite"
    rebuild_full=1
  fi

  if [[ "$rebuild_full" == "1" ]]; then
    echo "==> [$size] Converting language model to $outtype GGUF"
    rm -f "$full"
    python "$converter" "$source" --outfile "$full" --outtype "$outtype"
  else
    echo "==> [$size] Reusing $full"
  fi

  if [[ "$force" == "1" || ! -s "$mmproj" ]]; then
    echo "==> [$size] Converting multimodal projector to $outtype GGUF"
    rm -f "$mmproj"
    # Important: with a custom --outfile, llama.cpp b9637 writes exactly that
    # path; it does not prepend mmproj-. Pass the dedicated path explicitly so
    # the projector can never overwrite the language GGUF.
    python "$converter" "$source" --outfile "$mmproj" --outtype "$outtype" --mmproj
  else
    echo "==> [$size] Reusing $mmproj"
  fi

  if [[ ! -s "$full" || ! -s "$mmproj" ]]; then
    echo "error: GGUF conversion did not produce both model and mmproj for $size" >&2
    return 1
  fi

  local quant lower quant_file
  for quant in $quants; do
    lower="$(printf '%s' "$quant" | tr '[:upper:]' '[:lower:]')"
    quant_file="$out_dir/${base}-${lower}.gguf"
    if [[ "$force" == "1" || "$rebuild_full" == "1" || ! -s "$quant_file" ]]; then
      echo "==> [$size] Quantizing language model as $quant"
      rm -f "$quant_file"
      "$quantizer" "$full" "$quant_file" "$quant"
    else
      echo "==> [$size] Reusing $quant_file"
    fi
  done

  SIZE="$size" SOURCE="$source" FULL="$full" MMPROJ="$mmproj" \
  OUT_DIR="$out_dir" OUTTYPE="$outtype" QUANTS="$quants" \
  LLAMA_COMMIT="$llama_commit" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
size = os.environ["SIZE"]
base = f"gui-owl-{size}"
files = [Path(os.environ["FULL"]), Path(os.environ["MMPROJ"])]
for quant in os.environ["QUANTS"].split():
    files.append(out_dir / f"{base}-{quant.lower()}.gguf")

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "schema_version": 2,
    "student": size,
    "source_checkpoint": os.environ["SOURCE"],
    "llama_cpp_commit": os.environ["LLAMA_COMMIT"],
    "conversion_outtype": os.environ["OUTTYPE"],
    "quantizations": os.environ["QUANTS"].split(),
    "files": {
        path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in files
    },
}
path = out_dir / f"{base}-export-manifest.json"
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
PY
}

for size in "${sizes[@]}"; do
  do_convert "$size"
done

echo
printf 'GGUF export complete. llama.cpp commit: %s\n' "$llama_commit"
find "$out_dir" -maxdepth 1 -type f \( -name 'gui-owl-*.gguf' -o -name 'mmproj-gui-owl-*.gguf' -o -name '*-export-manifest.json' \) -print | sort
