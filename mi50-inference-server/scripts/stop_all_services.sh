#\!/usr/bin/env bash

# Colori
RED='''033[0;31m'''
GREEN='''033[0;32m'''
YELLOW='''033[1;33m'''
BLUE='''033[0;34m'''
NC='''033[0m'''

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  MI50 Stack - Stop Servizi${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Trova e killa tutti i processi
echo "🔍 Ricerca processi attivi..."
echo ""

# Backend MI50 (pt_main_t process)
MI50_PIDS=$(pgrep -f "pt_main_t" || true)
if [[ -n "$MI50_PIDS" ]]; then
    echo -e "${YELLOW}Backend MI50:${NC}"
    echo "  PID: $MI50_PIDS"
    kill $MI50_PIDS 2>/dev/null || true
    echo -e "  ${GREEN}✓ Terminato${NC}"
else
    echo "Backend MI50: non attivo"
fi
echo ""

# Chat UI
CHAT_PIDS=$(pgrep -f "uvicorn.*mi50_chat_ui" || true)
if [[ -n "$CHAT_PIDS" ]]; then
    echo -e "${YELLOW}Chat UI:${NC}"
    echo "  PID: $CHAT_PIDS"
    kill $CHAT_PIDS 2>/dev/null || true
    echo -e "  ${GREEN}✓ Terminato${NC}"
else
    echo "Chat UI: non attivo"
fi
echo ""

# Dashboard
DASH_PIDS=$(pgrep -f "uvicorn.*mi50_dashboard" || true)
if [[ -n "$DASH_PIDS" ]]; then
    echo -e "${YELLOW}Dashboard:${NC}"
    echo "  PID: $DASH_PIDS"
    kill $DASH_PIDS 2>/dev/null || true
    echo -e "  ${GREEN}✓ Terminato${NC}"
else
    echo "Dashboard: non attivo"
fi
echo ""

# Aspetta 2 secondi per terminazione graceful
sleep 2

# Force kill se ancora presenti
REMAINING=$(lsof -iTCP:11534,12000,13000 -sTCP:LISTEN 2>/dev/null || true)
if [[ -n "$REMAINING" ]]; then
    echo -e "${RED}⚠ Alcuni processi non si sono chiusi, forzo terminazione...${NC}"
    lsof -tiTCP:11534,12000,13000 -sTCP:LISTEN 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Tutti i servizi fermati${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Verifica finale
ACTIVE=$(lsof -iTCP:11534,12000,13000 -sTCP:LISTEN 2>/dev/null || true)
if [[ -z "$ACTIVE" ]]; then
    echo -e "${GREEN}✓ Nessun servizio in ascolto sulle porte 11534, 12000, 13000${NC}"
else
    echo -e "${RED}✗ Processi ancora attivi:${NC}"
    lsof -iTCP:11534,12000,13000 -sTCP:LISTEN
fi
echo ""
