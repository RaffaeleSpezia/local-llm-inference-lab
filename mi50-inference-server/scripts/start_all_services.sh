#\!/usr/bin/env bash
set -euo pipefail

# ==============================================
# Start All MI50 Services
# ==============================================
# Avvia in sequenza:
#   1. Backend MI50 (porta 11534)
#   2. Chat UI     (porta 12000)
#   3. Dashboard   (porta 13000)
# ==============================================

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="/mnt/raid0/shared_envs/venv-rocm311"
LOG_DIR="/tmp/mi50_services_logs"

# Colori
RED='''033[0;31m'''
GREEN='''033[0;32m'''
YELLOW='''033[1;33m'''
BLUE='''033[0;34m'''
NC='''033[0m''' # No Color

# Crea directory log
mkdir -p "$LOG_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  MI50 Stack - Avvio Servizi${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Funzione per verificare se una porta è occupata
check_port() {
    local port=$1
    if lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        return 0  # Occupata
    else
        return 1  # Libera
    fi
}

# Funzione per killare processo su porta
kill_port() {
    local port=$1
    local pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo -e "${YELLOW}  → Termino processi su porta $port: $pids${NC}"
        echo "$pids" | xargs kill 2>/dev/null || true
        sleep 2
    fi
}

# Funzione per aspettare che una porta sia attiva
wait_for_port() {
    local port=$1
    local max_wait=30
    local waited=0
    
    while \! check_port "$port"; do
        if (( waited >= max_wait )); then
            return 1
        fi
        sleep 1
        (( waited++ ))
    done
    return 0
}

# ==============================================
# 1. BACKEND MI50 (porta 11534)
# ==============================================
echo -e "${BLUE}[1/3] Backend MI50 LLM${NC}"
echo "      Porta: 11534"
echo "      Path: $BASE_DIR/mi50_come_ollama"

if check_port 11534; then
    echo -e "${YELLOW}  ⚠ Porta 11534 già occupata${NC}"
    read -p "  Vuoi killare il processo esistente? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        kill_port 11534
    else
        echo -e "${RED}  ✗ Salto avvio backend MI50${NC}"
        SKIP_MI50=1
    fi
fi

if [[ -z "${SKIP_MI50:-}" ]]; then
    cd "$BASE_DIR/mi50_come_ollama"
    
    # Avvia in background senza conferma interattiva
    source "$VENV_PATH/bin/activate"
    
    export HF_HOME="/mnt/raid0/hf_cache"
    export TRANSFORMERS_CACHE="/mnt/raid0/hf_cache/transformers"
    export TORCH_HOME="/mnt/raid0/torch_cache"
    export OLLAMA_FAKE_LOGDIR="/dev/shm/mi50_ollama_logs"
    export OLLAMA_FAKE_LOGLEVEL="info"
    export PYTHONPATH="$BASE_DIR/mi50_come_ollama"
    export HIP_VISIBLE_DEVICES="0"
    export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.95,expandable_segments:True"
    
    nohup python app.py "/mnt/raid0/qwen2.5-coder-7b-instruct" \
        --host 0.0.0.0 \
        --port 11534 \
        --log-level info \
        > "$LOG_DIR/mi50_backend.log" 2>&1 &
    
    MI50_PID=$\!
    echo "  → PID: $MI50_PID"
    echo "  → Log: $LOG_DIR/mi50_backend.log"
    
    # Aspetta che sia pronto
    echo -n "  → Attendo avvio..."
    if wait_for_port 11534; then
        echo -e " ${GREEN}✓${NC}"
    else
        echo -e " ${RED}✗ Timeout${NC}"
    fi
fi

echo ""

# ==============================================
# 2. CHAT UI (porta 12000)
# ==============================================
echo -e "${BLUE}[2/3] Chat UI${NC}"
echo "      Porta: 12000"
echo "      Path: $BASE_DIR/mi50_chat_ui"

if check_port 12000; then
    echo -e "${YELLOW}  ⚠ Porta 12000 già occupata${NC}"
    kill_port 12000
fi

cd "$BASE_DIR/mi50_chat_ui"
source "$VENV_PATH/bin/activate"

export MI50_SERVER_URL="http://127.0.0.1:11534"
export PATH=/opt/rocm/bin:$PATH

nohup uvicorn app.main:app --host 0.0.0.0 --port 12000 \
    > "$LOG_DIR/chat_ui.log" 2>&1 &

CHAT_PID=$\!
echo "  → PID: $CHAT_PID"
echo "  → Log: $LOG_DIR/chat_ui.log"

echo -n "  → Attendo avvio..."
if wait_for_port 12000; then
    echo -e " ${GREEN}✓${NC}"
else
    echo -e " ${RED}✗ Timeout${NC}"
fi

echo ""

# ==============================================
# 3. DASHBOARD (porta 13000)
# ==============================================
echo -e "${BLUE}[3/3] Dashboard${NC}"
echo "      Porta: 13000"
echo "      Path: $BASE_DIR/mi50_dashboard"

if check_port 13000; then
    echo -e "${YELLOW}  ⚠ Porta 13000 già occupata${NC}"
    kill_port 13000
fi

cd "$BASE_DIR/mi50_dashboard"
source "$VENV_PATH/bin/activate"

export MI50_SERVER_URL="http://127.0.0.1:11534"
export PATH=/opt/rocm/bin:$PATH

nohup uvicorn app.main:app --host 0.0.0.0 --port 13000 \
    > "$LOG_DIR/dashboard.log" 2>&1 &

DASH_PID=$\!
echo "  → PID: $DASH_PID"
echo "  → Log: $LOG_DIR/dashboard.log"

echo -n "  → Attendo avvio..."
if wait_for_port 13000; then
    echo -e " ${GREEN}✓${NC}"
else
    echo -e " ${RED}✗ Timeout${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Tutti i servizi avviati\!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "📊 Status:"
echo ""
lsof -iTCP:11534,12000,13000 -sTCP:LISTEN 2>/dev/null || echo "  Nessun servizio in ascolto"
echo ""
echo "🌐 URL:"
echo "  • Backend MI50: http://192.168.1.155:11534/api/version"
echo "  • Chat UI:      http://192.168.1.155:12000"
echo "  • Dashboard:    http://192.168.1.155:13000"
echo ""
echo "📋 Log:"
echo "  • Backend: tail -f $LOG_DIR/mi50_backend.log"
echo "  • Chat UI: tail -f $LOG_DIR/chat_ui.log"
echo "  • Dashboard: tail -f $LOG_DIR/dashboard.log"
echo ""
echo "🛑 Stop all:"
echo "  pkill -f 'pt_main_t|uvicorn.*mi50'"
echo ""
