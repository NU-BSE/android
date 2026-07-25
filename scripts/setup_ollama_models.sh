#!/usr/bin/env bash
set -euo pipefail

export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama CLI is required" >&2
  exit 2
fi

ollama pull qwen3-vl:30b
ollama pull avil/UI-TARS:latest
ollama create android-qwen3-vl-teacher -f ollama/Modelfile.qwen3-vl-teacher
ollama create android-ui-tars-critic -f ollama/Modelfile.ui-tars-critic

python scripts/check_ollama_models.py --config config/ollama_models.json
