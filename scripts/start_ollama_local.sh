#!/usr/bin/env bash
set -euo pipefail

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama CLI is required" >&2
  exit 2
fi

export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
exec ollama serve
