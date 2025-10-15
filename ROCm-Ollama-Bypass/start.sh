#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VENV="$BASE_DIR/.venv"
DEFAULT_MODEL="$BASE_DIR/sample-model"
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="11534"
DEFAULT_LOGDIR="$BASE_DIR/logs"

VENV_PATH="${VENV_PATH:-$DEFAULT_VENV}"
MODEL_PATH="${1:-${OLLAMA_FAKE_DEFAULT_MODEL:-$DEFAULT_MODEL}}"
BIND_HOST="${HOST:-$DEFAULT_HOST}"
BIND_PORT="${PORT:-$DEFAULT_PORT}"
LOG_LEVEL="${OLLAMA_FAKE_LOGLEVEL:-info}"
LOG_DIR="${OLLAMA_FAKE_LOGDIR:-$DEFAULT_LOGDIR}"
HF_HOME_VALUE="${HF_HOME:-$BASE_DIR/.cache/hf}"
TRANSFORMERS_CACHE_VALUE="${TRANSFORMERS_CACHE:-$BASE_DIR/.cache/hf/transformers}"
TORCH_HOME_VALUE="${TORCH_HOME:-$BASE_DIR/.cache/torch}"

cat <<INTRO
=============================================
 MI50 Ollama helper (start.sh)
=============================================
Percorsi chiave:
  - Servizio : $BASE_DIR
  - Venv     : $VENV_PATH
  - Modello  : $MODEL_PATH
  - Log JSON : $LOG_DIR/mi50_ollama.log

Cache e log verranno scritti nelle directory riportate qui sotto:
  HF_HOME=$HF_HOME_VALUE
  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE_VALUE
  TORCH_HOME=$TORCH_HOME_VALUE

Avvio manuale:
  source $VENV_PATH/bin/activate
  export HF_HOME=$HF_HOME_VALUE
  export TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE_VALUE
  export TORCH_HOME=$TORCH_HOME_VALUE
  export OLLAMA_FAKE_LOGDIR=$LOG_DIR
  export OLLAMA_FAKE_LOGLEVEL=$LOG_LEVEL
  python app.py "$MODEL_PATH" --host $BIND_HOST --port $BIND_PORT --log-level $LOG_LEVEL

Per eseguire i test:
  source $VENV_PATH/bin/activate
  PYTHONPATH=$BASE_DIR pytest tests/test_app.py -q

Per il deploy continuo:
  sudo cp systemd/mi50_ollama.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable --now mi50_ollama.service
  journalctl -u mi50_ollama.service -f

INTRO

read -rp "Vuoi avviare ora il server con queste impostazioni? [y/N] " answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
  if [[ ! -d "$VENV_PATH" ]]; then
    echo "ERRORE: ambiente virtuale non trovato in $VENV_PATH" >&2
    exit 1
  fi

  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "AVVISO: $MODEL_PATH non è una directory locale, verrà usato come nome modello remoto." >&2
  fi

  if lsof -tiTCP:"$BIND_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "ERRORE: la porta $BIND_PORT risulta già in uso (processo visibile via lsof)." >&2
    exit 1
  fi

  if ss -ltn "sport = :$BIND_PORT" 2>/dev/null | tail -n +2 | grep -q ":$BIND_PORT"; then
    echo "ERRORE: la porta $BIND_PORT risulta occupata (rilevato da ss)." >&2
    echo "Suggerimento: libera la porta (es. sudo systemctl stop ollama) oppure avvia con PORT=<nuova porta>." >&2
    exit 1
  fi

  mkdir -p "$LOG_DIR"
  source "$VENV_PATH/bin/activate"
  export HF_HOME="$HF_HOME_VALUE"
  export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE_VALUE"
  export TORCH_HOME="$TORCH_HOME_VALUE"
  export OLLAMA_FAKE_LOGDIR="$LOG_DIR"
  export OLLAMA_FAKE_LOGLEVEL="$LOG_LEVEL"
  export PYTHONPATH="$BASE_DIR"

  echo "\n[MI50 Ollama] Avvio in corso..."
  echo "       Modello : $MODEL_PATH"
  echo "       Host    : $BIND_HOST"
  echo "       Porta   : $BIND_PORT"
  echo "       Log     : $LOG_DIR/mi50_ollama.log"
  echo "       Venv    : $VENV_PATH"
  echo
  exec python "$BASE_DIR/app.py" "$MODEL_PATH" --host "$BIND_HOST" --port "$BIND_PORT" --log-level "$LOG_LEVEL"
else
  cat <<'SKIP'

Avvio rimandato. Puoi rieseguire questo script quando vuoi, oppure seguire le istruzioni riportate qui sopra.
Per uscire: Ctrl+C
SKIP
fi
