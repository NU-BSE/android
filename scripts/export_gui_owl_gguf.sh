#!/usr/bin/env bash
set -euo pipefail

# Convert one or all accepted v2 GUI-Owl students to llama.cpp GGUF + mmproj.
# The source checkpoint is resolved from config/gui_owl_students.json so the
# exporter cannot accidentally pick an older collapsed checkpoint.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
student_selector="${1:-${STUDENT_SIZE:-all}}"
student_config="${STUDENT_CONFIG:-$repo_root/config/gui_owl_students.json}"
llama_cpp_dir="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
llama_cpp_ref="${LLAMA_CPP_REF:-b9637}"
out_dir="${GGUF_OUT_DIR:-$repo_root/artifacts/gguf/v2}"
outtype="${GGUF_OUTTYPE:-bf16}"
quants="${GGUF_QUANTS:-Q4_0}"
force="${GGUF_FORCE:-0}"
allow_unaccepted="${GGUF_ALLOW_UNACCEPTED:-0}"
install_deps="${LLAMA_CPP_INSTALL_DEPS:-0}"
build_quantizer="${LLAMA_CPP_BUILD_QUANTIZER:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/export_gui_owl_gguf.sh [0.5b|1b|1.5b|all]

Environment:
  STUDENT_CONFIG             student config JSON
  LLAMA_CPP_DIR              llama.cpp checkout (default: ~/llama.cpp)
  LLAMA_CPP_REF              ref cloned when checkout is absent (default: b9637)
  LLAMA_CPP_INSTALL_DEPS=1   allow pip install -r llama.cpp/requirements.txt
  LLAMA_CPP_BUILD_QUANTIZER  build llama-quantize when missing (default: 1)
  GGUF_OUT_DIR               output directory (default: artifacts/gguf/v2)
  GGUF_OUTTYPE               conversion precision (default: bf16)
  GGUF_QUANTS                space-separated quantizations (default: Q4_0)
  GGUF_FORCE=1               overwrite existing GGUF files
  GGUF_ALLOW_UNACCEPTED=1    debug-only override for a failed generation gate

By default export requires distillation-manifest.json with accepted=true. The
multimodal projector remains at GGUF_OUTTYPE precision; only the language model
is quantized.
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

resolve_source() {
  local size="$1"
  python - "$student_config" "$size" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
entry = config.get("students", {}).get(sys.argv[2], {})
path = entry.get("distilled")
if not path:
    raise SystemExit(f"student config lacks distilled path for {sys.argv[2]}")
print(path)
PY
}

check_acceptance() {
  local source="$1"
  if [[ "$allow_unaccepted" == "1" ]]; then
    return 0
  fi
  python - "$source/distillation-manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"refusing GGUF export: distillation manifest missing: {path}")
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("accepted") is not True:
    failures = value.get("final_gate_failures") or []
    raise SystemExit(
        "refusing GGUF export: student did not pass free-generation gates"
        + (f" ({'; '.join(map(str, failures))})" if failures else "")
    )
PY
}

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
    python -m pip install -r "$llama_cpp_dir/requirements.txt"
  else
    echo "error: llama.cpp conversion dependencies are missing" >&2
    echo "run: python -m pip install -r '$llama_cpp_dir/requirements.txt'" >&2
    exit 1
  fi
fi

quantizer="$llama_cpp_dir/build/bin/llama-quantize"
if [[ ! -x "$quantizer" ]]; then
  if [[ "$build_quantizer" != "1" ]]; then
    echo "error: llama-quantize is missing: $quantizer" >&2
    exit 1
  fi
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
  local source_rel source base full mmproj rebuild_full
  source_rel="$(resolve_source "$size")"
  if [[ "$source_rel" = /* ]]; then
    source="$source_rel"
  else
    source="$repo_root/$source_rel"
  fi
  base="gui-owl-${size}"
  full="$out_dir/${base}-${outtype}.gguf"
  mmproj="$out_dir/mmproj-${base}-${outtype}.gguf"
  rebuild_full=0

  if [[ ! -f "$source/config.json" ]]; then
    echo "error: distilled $size checkpoint is missing: $source/config.json" >&2
    return 1
  fi
  if ! compgen -G "$source/*.safetensors" >/dev/null; then
    echo "error: distilled $size checkpoint has no safetensors weights: $source" >&2
    return 1
  fi
  check_acceptance "$source"

  if [[ "$force" == "1" || ! -s "$full" ]]; then
    rebuild_full=1
  elif [[ ! -s "$mmproj" ]]; then
    echo "==> [$size] mmproj is missing; rebuilding language GGUF too"
    rebuild_full=1
  fi

  if [[ "$rebuild_full" == "1" ]]; then
    echo "==> [$size] Converting accepted language model to $outtype GGUF"
    rm -f "$full"
    python "$converter" "$source" --outfile "$full" --outtype "$outtype"
  else
    echo "==> [$size] Reusing $full"
  fi

  if [[ "$force" == "1" || ! -s "$mmproj" ]]; then
    echo "==> [$size] Converting multimodal projector to $outtype GGUF"
    rm -f "$mmproj"
    local mmproj_tmp mmproj_candidate mmproj_count
    mmproj_tmp="$(mktemp -d "$out_dir/.mmproj-${base}.XXXXXX")"
    if ! python "$converter" "$source" --outfile "$mmproj_tmp" --outtype "$outtype" --mmproj; then
      rm -rf "$mmproj_tmp"
      return 1
    fi
    mmproj_count="$(find "$mmproj_tmp" -maxdepth 1 -type f -name '*.gguf' | wc -l | tr -d '[:space:]')"
    if [[ "$mmproj_count" != "1" ]]; then
      echo "error: expected exactly one mmproj GGUF for $size, found $mmproj_count" >&2
      find "$mmproj_tmp" -maxdepth 1 -type f -print >&2 || true
      rm -rf "$mmproj_tmp"
      return 1
    fi
    mmproj_candidate="$(find "$mmproj_tmp" -maxdepth 1 -type f -name '*.gguf' -print -quit)"
    mv "$mmproj_candidate" "$mmproj"
    rm -rf "$mmproj_tmp"
  else
    echo "==> [$size] Reusing $mmproj"
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
    "schema_version": 4,
    "student": size,
    "source_checkpoint": os.environ["SOURCE"],
    "source_generation_gate_required": True,
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

printf '\nGGUF export complete. llama.cpp commit: %s\n' "$llama_commit"
find "$out_dir" -maxdepth 1 -type f \( \
  -name 'gui-owl-*.gguf' -o \
  -name 'mmproj-gui-owl-*.gguf' -o \
  -name '*-export-manifest.json' \
\) -print | sort
