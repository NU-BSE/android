#!/usr/bin/env bash
set -euo pipefail

# Run on macOS after the GPU host packages accepted v2 student bundles.

host="${VAST_HOST:-}"
port="${VAST_PORT:-}"
user="${VAST_USER:-root}"
remote_root="${VAST_REMOTE_REPO:-/root/android}"
remote_transfer_rel="${REMOTE_TRANSFER_DIR:-artifacts/transfer/v2}"
dest="${GUI_OWL_MAC_DEST:-$HOME/Models/gui-owl-v2}"
student_selector="all"

usage() {
  cat <<'EOF'
Usage (run on macOS):
  bash scripts/pull_gui_owl_students_to_mac.sh \
    --host <VAST_HOST> --port <VAST_SSH_PORT> [options]

Options:
  --host HOST          Vast SSH host/IP (or set VAST_HOST)
  --port PORT          Vast external SSH port (or set VAST_PORT)
  --user USER          SSH user (default: root)
  --remote-root PATH   repo path on Vast (default: /root/android)
  --dest PATH          local destination (default: ~/Models/gui-owl-v2)
  --student SIZE       0.5b, 1b, 1.5b, or all (default: all)
  -h, --help           show this help

The remote host must first run:
  bash scripts/package_gui_owl_students.sh all

By default bundles are read from artifacts/transfer/v2. The script uses rsync
with --partial when available on both hosts, otherwise scp. It never deletes
remote files or stops the GPU instance.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) host="${2:?missing --host value}"; shift 2 ;;
    --port) port="${2:?missing --port value}"; shift 2 ;;
    --user) user="${2:?missing --user value}"; shift 2 ;;
    --remote-root) remote_root="${2:?missing --remote-root value}"; shift 2 ;;
    --dest) dest="${2:?missing --dest value}"; shift 2 ;;
    --student) student_selector="${2:?missing --student value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$host" || -z "$port" ]]; then
  echo "error: --host and --port (or VAST_HOST/VAST_PORT) are required" >&2
  usage >&2
  exit 2
fi
case "$student_selector" in
  0.5b|1b|1.5b) sizes=("$student_selector") ;;
  all) sizes=(0.5b 1b 1.5b) ;;
  *) echo "error: invalid student size: $student_selector" >&2; exit 2 ;;
esac
if ! [[ "$port" =~ ^[0-9]+$ ]]; then
  echo "error: SSH port must be numeric" >&2
  exit 2
fi

for command in ssh scp tar shasum; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "error: required macOS command not found: $command" >&2
    exit 1
  fi
done

mkdir -p "$dest/downloads" "$dest/extracted"
remote="${user}@${host}"
ssh_args=(-p "$port" -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
ssh_transport="ssh -p $port -o ServerAliveInterval=30 -o ServerAliveCountMax=6"

use_rsync=0
if command -v rsync >/dev/null 2>&1; then
  if ssh "${ssh_args[@]}" "$remote" 'command -v rsync >/dev/null 2>&1' </dev/null; then
    use_rsync=1
  fi
fi

fetch_file() {
  local remote_path="$1"
  local local_dir="$2"
  if [[ "$use_rsync" == "1" ]]; then
    rsync --partial --progress -e "$ssh_transport" "$remote:$remote_path" "$local_dir/"
  else
    scp -P "$port" \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=6 \
      "$remote:$remote_path" "$local_dir/"
  fi
}

for size in "${sizes[@]}"; do
  base="gui-owl-${size}-mac-bundle.tar"
  remote_dir="$remote_root/$remote_transfer_rel"
  echo "==> [$size] Downloading v2 bundle"
  fetch_file "$remote_dir/$base" "$dest/downloads"
  fetch_file "$remote_dir/$base.sha256" "$dest/downloads"

  echo "==> [$size] Verifying SHA-256"
  (
    cd "$dest/downloads"
    shasum -a 256 -c "$base.sha256"
  )

  echo "==> [$size] Extracting"
  mkdir -p "$dest/extracted/$size"
  tar -xf "$dest/downloads/$base" -C "$dest/extracted/$size"
done

printf '\nAll requested v2 bundles verified and extracted under:\n  %s\n' "$dest/extracted"
printf 'Keep the .tar files until local GGUF/mmproj inference testing passes.\n'
