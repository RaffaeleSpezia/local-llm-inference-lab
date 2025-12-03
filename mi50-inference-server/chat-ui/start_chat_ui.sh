#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-/mnt/raid0/shared_envs/venv-rocm311}"
INFERENCE_URL="${MI50_SERVER_URL:-http://127.0.0.1:11534}"
PORT="${CHAT_UI_PORT:-13010}"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Venv non trovato in $VENV_PATH" >&2
  exit 1
fi

# Kill any uvicorn already binding this port
mapfile -t old_pids < <(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
if (( ${#old_pids[@]} > 0 )); then
  echo "[INFO] Porto $PORT occupata da PID: ${old_pids[*]} → termino..."
  for pid in "${old_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  sleep 1
fi

source "$VENV_PATH/bin/activate"
export PATH=/opt/rocm/bin:$PATH
pip install -r "$BASE_DIR/requirements.txt" >/dev/null

export MI50_SERVER_URL="$INFERENCE_URL"
cd "$BASE_DIR"
uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
