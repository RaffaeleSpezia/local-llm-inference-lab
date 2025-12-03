#\!/usr/bin/env bash

# Colori
RED='''033[0;31m'''
GREEN='''033[0;32m'''
YELLOW='''033[1;33m'''
BLUE='''033[0;34m'''
NC='''033[0m'''

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  MI50 Stack - Status Servizi${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Funzione per check porta
check_service() {
    local name=$1
    local port=$2
    local url=$3
    
    echo -e "${YELLOW}$name (porta $port):${NC}"
    
    # Check processo
    local pid=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
    if [[ -n "$pid" ]]; then
        echo -e "  Status: ${GREEN}✓ Running${NC}"
        echo "  PID: $pid"
        
        # Test HTTP se URL fornito
        if [[ -n "$url" ]]; then
            if curl -s -f -m 2 "$url" >/dev/null 2>&1; then
                echo -e "  HTTP: ${GREEN}✓ Responding${NC}"
            else
                echo -e "  HTTP: ${RED}✗ Not responding${NC}"
            fi
        fi
    else
        echo -e "  Status: ${RED}✗ Not running${NC}"
    fi
    echo ""
}

# Check servizi
check_service "Backend MI50" 11534 "http://127.0.0.1:11534/api/version"
check_service "Chat UI" 12000 "http://127.0.0.1:12000/"
check_service "Dashboard" 13000 "http://127.0.0.1:13000/"

# Porte in ascolto
echo -e "${BLUE}Porte in ascolto:${NC}"
lsof -iTCP:11534,12000,13000 -sTCP:LISTEN 2>/dev/null || echo "  Nessuna"
echo ""

# VRAM status
echo -e "${BLUE}VRAM MI50:${NC}"
curl -s http://127.0.0.1:11534/debug/memory 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  Backend offline"
echo ""

# URL accesso
echo -e "${BLUE}URL Accesso:${NC}"
echo "  • Backend: http://192.168.1.155:11534/api/version"
echo "  • Chat UI: http://192.168.1.155:12000"
echo "  • Dashboard: http://192.168.1.155:13000"
echo ""
