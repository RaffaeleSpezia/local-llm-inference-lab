# Dashboard Setup - Monitoring & Admin UI

**Versione:** 1.0
**Data:** 3 Dicembre 2025
**Autore:** Raffaele Spezia

---

## Indice

1. [Overview](#overview)
2. [Installazione](#installazione)
3. [Features](#features)
4. [Avvio](#avvio)
5. [Customizzazione](#customizzazione)

---

## Overview

**Dashboard** è un'interfaccia web per monitoring del sistema MI50 Stack.

**Features:**
- VRAM monitoring real-time (MI50)
- Model status + info
- Service health checks
- Token/second metrics
- Request history
- System resources (CPU, RAM, GPU)

**Stack:**
- Flask 3.0 (web framework)
- Requests (HTTP client)
- HTML/JS/CSS (frontend)
- Optional: Streamlit per dashboar

d avanzate

---

## Installazione

### Setup Base

```bash
ssh lele2@192.168.1.155
cd ~/mi50_stack

# Crea directory
mkdir -p mi50_dashboard
cd mi50_dashboard

# Usa venv condiviso
source /mnt/raid0/shared_envs/venv-rocm311/bin/activate

# Install dependencies
pip install flask==3.0.3 requests==2.32.3
```

### File Structure

```
mi50_dashboard/
├── dashboard.py            # Flask app (~300 righe)
├── templates/
│   └── index.html          # Dashboard UI
├── start_dashboard.sh      # Avvio script
└── requirements.txt
```

### requirements.txt

```txt
flask==3.0.3
requests==2.32.3
```

---

## Features

### 1. VRAM Monitoring

**Display:**
```
MI50 VRAM Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Allocated: 14.2 GB / 32.0 GB (44%)
Reserved:  15.5 GB / 32.0 GB (48%)
Free:      16.5 GB
Waste:     1.3 GB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Endpoint:** `GET http://192.168.1.155:11534/debug/memory`

**Auto-refresh:** 2s interval

### 2. Model Status

**Display:**
```
Loaded Model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name: qwen2.5-coder-7b-instruct
Parameters: 7.6B
Context: 32768 tokens
Loaded: 2025-12-03 10:30:45
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Endpoint:** `GET http://192.168.1.155:11534/api/tags`

### 3. Service Health

**Checks:**
- Backend MI50 (11534)
- RAG M40 (11600)
- Chat UI (12000)
- Dashboard (13000)

**Display:**
```
Services Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Backend MI50   (11534) - OK
✓ RAG M40        (11600) - OK
✓ Chat UI        (12000) - OK
✓ Dashboard      (13000) - OK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 4. Performance Metrics

**Display:**
```
Generation Stats
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Last request: 85 tokens/sec
Avg latency: 2.3s
Total requests: 142
Failed: 0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Avvio

### Manuale

```bash
cd ~/mi50_stack/mi50_dashboard
source /mnt/raid0/shared_envs/venv-rocm311/bin/activate
export BACKEND_URL="http://127.0.0.1:11534"
python dashboard.py
```

### Via Script

```bash
cd ~/mi50_stack/mi50_dashboard
./start_dashboard.sh
```

**start_dashboard.sh:**
```bash
#!/bin/bash
source /mnt/raid0/shared_envs/venv-rocm311/bin/activate
export BACKEND_URL="http://127.0.0.1:11534"
PORT=${DASHBOARD_PORT:-13000}
python dashboard.py --port $PORT
```

### Via start_all_services.sh

```bash
cd ~/mi50_stack
./start_all_services.sh
```

### Accesso

**Browser:**
```
http://192.168.1.155:13000
```

---

## Customizzazione

### Aggungere Metric

**In dashboard.py:**

```python
@app.route('/api/custom_metric')
def custom_metric():
    # Fetch data
    data = requests.get(f"{BACKEND_URL}/debug/memory").json()

    # Calculate metric
    metric_value = data['allocated_gb'] / data['total_gb'] * 100

    return jsonify({"metric": metric_value})
```

**In templates/index.html:**

```javascript
async function updateCustomMetric() {
    const response = await fetch('/api/custom_metric');
    const data = await response.json();
    document.getElementById('custom-metric').innerText = data.metric.toFixed(2) + '%';
}
setInterval(updateCustomMetric, 5000);
```

---

## Next Steps

**→ [06-systemd-services.md](./06-systemd-services.md)** - Systemd service configuration

---

**Licenza:** CC BY-NC-SA 4.0
**Autore:** Raffaele Spezia
**Repository:** https://github.com/RaffaeleSpezia/local-llm-inference-lab
