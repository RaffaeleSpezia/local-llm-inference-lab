#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/mi50_come_ollama"
START_SCRIPT="$SERVER_DIR/start.sh"
DEFAULT_MODEL_KEY="qwen2.5-coder-7b"
DEFAULT_HOST="${HOST:-0.0.0.0}"
DEFAULT_PORT="${PORT:-11534}"
DEFAULT_LOG_LEVEL="${OLLAMA_FAKE_LOGLEVEL:-info}"
DEFAULT_LOG_DIR="${OLLAMA_FAKE_LOGDIR:-/dev/shm/mi50_ollama_logs}"
DEFAULT_HF_HOME="${HF_HOME:-/mnt/raid0/hf_cache}"
DEFAULT_TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/raid0/hf_cache/transformers}"
DEFAULT_TORCH_HOME="${TORCH_HOME:-/mnt/raid0/torch_cache}"

# Registry alias -> model path
declare -A MODEL_PATHS=(
  [qwen2.5-coder-7b]="/mnt/raid0/qwen2.5-coder-7b-instruct"
  [qwen2.5-coder-14b]="/mnt/raid0/qwen2.5-coder-14b-instruct"
  [gemma-2-9b-it]="/mnt/raid0/gemma-2-9b-it"
  [gemma-3-12b-it]="/mnt/raid0/gemma-3-12b-it"
  [gemma3-4b-it]="/mnt/raid0/gemma3-4b-it"
  [deepseek-coder-6.7b]="/mnt/raid0/deepseek-coder-6.7b-instruct"
)

declare -A MODEL_NOTES=(
  [qwen2.5-coder-7b]="Qwen2.5 Coder 7B Instruct FP16 – consigliato (50-100 tok/s su MI50)."
  [qwen2.5-coder-14b]="Qwen2.5 Coder 14B Instruct FP16 – più qualità, usa ~28GB VRAM."
  [gemma-2-9b-it]="Gemma 2 9B Instruct – bilanciato qualità/VRAM, buono per codice."
  [gemma-3-12b-it]="Gemma 3 12B Italian – ~23GB VRAM, qualità superiore."
  [gemma3-4b-it]="Gemma 3 4B Italian – ~8GB VRAM, italiana, leggero."
  [deepseek-coder-6.7b]="DeepSeek Coder 6.7B Instruct – ottimizzato per coding, ~13GB VRAM."
)

declare -A MODEL_VRAM=(
  [qwen2.5-coder-7b]="~14GB"
  [qwen2.5-coder-14b]="~28GB"
  [gemma-2-9b-it]="~18GB"
  [gemma-3-12b-it]="~23GB"
  [gemma3-4b-it]="~8GB"
  [deepseek-coder-6.7b]="~13GB"
)

MODEL_ORDER=(qwen2.5-coder-7b qwen2.5-coder-14b gemma-2-9b-it gemma-3-12b-it deepseek-coder-6.7b gemma3-4b-it)

kill_running_backend() {
  local pids
  pids=$(ps -eo pid,cmd | awk '/mi50_come_ollama\/app.py/ {print $1}')
  if [[ -z "$pids" ]]; then
    return
  fi
  echo "[INFO] Backend già attivo (PID: $pids) → invio SIGTERM"
  kill $pids 2>/dev/null || true
  for attempt in {1..5}; do
    sleep 2
    pids=$(ps -eo pid,cmd | awk '/mi50_come_ollama\/app.py/ {print $1}')
    [[ -z "$pids" ]] && break
    echo "[INFO] Attendo arresto backend (tentativo $attempt/5)..."
  done
  if [[ -n "$pids" ]]; then
    echo "[INFO] Backend ancora vivo → invio SIGKILL"
    kill -9 $pids 2>/dev/null || true
    sleep 2
  fi
}

usage() {
  cat <<USAGE
Usage: $(basename "$0") [list|MODEL_KEY] [--port <PORT>] [--host <HOST>] [--log-level <LEVEL>]

Comandi:
  list             Mostra i modelli registrati.
  MODEL_KEY        Alias del modello da caricare (default $DEFAULT_MODEL_KEY).

Opzioni:
  --port <PORT>        Porta HTTP (default ${DEFAULT_PORT}).
  --host <HOST>        Host di bind (default ${DEFAULT_HOST}).
  --log-level <LEVEL>  Livello log FastAPI (default ${DEFAULT_LOG_LEVEL}).
USAGE
}

print_models() {
  printf "%-20s %-36s %-8s %s\n" "Alias" "Percorso" "VRAM" "Note"
  echo "----------------------------------------------------------------------------------------------"
  for key in "${MODEL_ORDER[@]}"; do
    printf "%-20s %-36s %-8s %s\n" "$key" "${MODEL_PATHS[$key]}" "${MODEL_VRAM[$key]}" "${MODEL_NOTES[$key]}"
  done
}

is_port_busy() {
  local port="$1"
  ss -ltn "sport = :$port" 2>/dev/null | tail -n +2 | grep -q ":$port"
}

ensure_port_free() {
  local port="$1"
  local pids
  pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "[INFO] Porta $port occupata dai PID: $pids → termino..."
    kill $pids 2>/dev/null || true
  fi
  for attempt in {1..5}; do
    if ! is_port_busy "$port"; then
      return
    fi
    echo "[INFO] Porta $port ancora occupata, attendo il rilascio ($attempt/5)..."
    sleep 2
  done
  echo "ERRORE: impossibile liberare la porta $port, arresto." >&2
  exit 1
}

MODEL_KEY=""
PORT="$DEFAULT_PORT"
HOST="$DEFAULT_HOST"
LOG_LEVEL="$DEFAULT_LOG_LEVEL"

if [[ $# -eq 0 ]]; then
  usage
  exit 0
fi

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    list)
      if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
        echo "Errore: list non si combina con altre opzioni." >&2
        exit 1
      fi
      print_models
      exit 0
      ;;
    --port)
      shift
      PORT="${1:-}"
      [[ -n "$PORT" ]] || { echo "--port richiede un argomento" >&2; exit 1; }
      ;;
    --host)
      shift
      HOST="${1:-}"
      [[ -n "$HOST" ]] || { echo "--host richiede un argomento" >&2; exit 1; }
      ;;
    --log-level)
      shift
      LOG_LEVEL="${1:-}"
      [[ -n "$LOG_LEVEL" ]] || { echo "--log-level richiede un argomento" >&2; exit 1; }
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Opzione sconosciuta: $1" >&2
      usage
      exit 1
      ;;
    *)
      POSITIONAL+=("$1")
      ;;
  esac
  shift
done

if [[ ${#POSITIONAL[@]} -eq 0 ]]; then
  MODEL_KEY="$DEFAULT_MODEL_KEY"
else
  MODEL_KEY="${POSITIONAL[0]}"
fi

if [[ -z "${MODEL_PATHS[$MODEL_KEY]:-}" ]]; then
  echo "Modello non registrato. Usa list per gli alias disponibili." >&2
  exit 1
fi

MODEL_PATH="${MODEL_PATHS[$MODEL_KEY]}"
MODEL_NOTE="${MODEL_NOTES[$MODEL_KEY]}"
MODEL_VRAM_REQ="${MODEL_VRAM[$MODEL_KEY]}"
LOG_DIR="$DEFAULT_LOG_DIR"
HF_HOME_VALUE="$DEFAULT_HF_HOME"
TRANSFORMERS_CACHE_VALUE="$DEFAULT_TRANSFORMERS_CACHE"
TORCH_HOME_VALUE="$DEFAULT_TORCH_HOME"

if [[ ! -x "$START_SCRIPT" ]]; then
  echo "Script di avvio non trovato in $START_SCRIPT" >&2
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[AVVISO] $MODEL_PATH non è una directory locale: verrà trattato come repo HuggingFace." >&2
fi

echo "[INFO] Controllo porta $PORT..."
ensure_port_free "$PORT"
kill_running_backend

cat <<INFO
=============================================
 MI50 AI Server Launcher
=============================================
Alias scelto  : $MODEL_KEY
Percorso/Repo : $MODEL_PATH
Descrizione   : $MODEL_NOTE
VRAM richiesta: $MODEL_VRAM_REQ
Host / Porta  : $HOST / $PORT
Log JSON      : $LOG_DIR/mi50_ollama.log
=============================================
INFO

echo "Suggerimento: per evitare doppi caricamenti usa i client senza campo model (useranno il default)."

MODEL_PATH_LOWER="${MODEL_PATH,,}"
if [[ $MODEL_PATH_LOWER == *"gemma-3-"* ]] && [[ -z ${OLLAMA_FORCE_DTYPE:-} ]]; then
  export OLLAMA_FORCE_DTYPE="bfloat16"
fi

cd "$SERVER_DIR"
HF_HOME="$HF_HOME_VALUE" \
TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE_VALUE" \
TORCH_HOME="$TORCH_HOME_VALUE" \
OLLAMA_FAKE_LOGDIR="$LOG_DIR" \
OLLAMA_FAKE_LOGLEVEL="$LOG_LEVEL" \
OLLAMA_FAKE_DEFAULT_MODEL="$MODEL_PATH" \
HOST="$HOST" PORT="$PORT" \
  printf "y\n" | "$START_SCRIPT" "$MODEL_PATH"
