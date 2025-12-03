# MI50 Dashboard

Pannello web leggero per visualizzare in tempo reale lo stato del servizio `mi50_come_ollama` e l'output token-per-token.

## Funzionalità
- Grafico VRAM allocata/riservata (lettura dall'endpoint `/debug/memory` del server PyTorch)
- Card CPU/RAM host + GPU (temperatura, utilizzo, VRAM libera)
- Streaming live dei token via `/ws/tokens` del server MI50 (solo lettura)
- Monitoraggio prompt/parametri e stato richieste senza influire sull'inferenza

## Requisiti
- Python 3.10+
- Venv ROCm esistente (default `/mnt/raid0/shared_envs/venv-rocm311`)
- Server MI50 attivo su `http://127.0.0.1:11534`
- `rocm-smi` disponibile nel PATH per raccogliere temperature/VRAM

## Avvio rapido
```bash
cd ~/mi50_stack/mi50_dashboard
./start_dashboard.sh                     # avvia su http://0.0.0.0:13000
# varianti:
DASHBOARD_PORT=8080 ./start_dashboard.sh
MI50_SERVER_URL=http://localhost:11534 ./start_dashboard.sh
```

Visita `http://<server>:13000` e:
1. verifica lo stato (dot verde = connesso a `/ws/metrics`)
2. inserisci il prompt, eventuali `max_new_tokens`/`temperature`
3. segui la risposta in streaming nella sezione "Risposta live"

## Struttura
- `app/main.py`: FastAPI + websocket per metriche e generazione
- `app/metrics.py`: poller asincrono CPU/RAM + `/debug/memory`
- `static/index.html`: UI (Chart.js + WebSocket client)
- `start_dashboard.sh`: helper per attivare il venv, installare dipendenze e lanciare Uvicorn
