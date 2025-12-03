#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-/mnt/raid0/shared_envs/venv-rocm311}"
INFERENCE_URL="${MI50_SERVER_URL:-http://127.0.0.1:11534}"
PORT="${DASHBOARD_PORT:-13000}"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Venv non trovato in $VENV_PATH" >&2
  exit 1
fi

source "$VENV_PATH/bin/activate"
export PATH=/opt/rocm/bin:$PATH
pip install -r "$BASE_DIR/requirements.txt" >/dev/null

export MI50_SERVER_URL="$INFERENCE_URL"
cd "$BASE_DIR"
uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
