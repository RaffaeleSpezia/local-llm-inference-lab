# Systemd Services - Auto-Start Configuration

**Versione:** 1.0
**Data:** 3 Dicembre 2025
**Autore:** Raffaele Spezia

---

## Indice

1. [Overview](#overview)
2. [Backend MI50 Service](#backend-mi50-service)
3. [Chat UI Service](#chat-ui-service)
4. [Dashboard Service](#dashboard-service)
5. [Multi-Service Management](#multi-service-management)

---

## Overview

Systemd permette di:
- Auto-start servizi al boot
- Restart automatico su crash
- Logging centralizzato (journalctl)
- Dependency management

---

## Backend MI50 Service

### Service File

**Path:** `/etc/systemd/system/mi50-backend.service`

```ini
[Unit]
Description=MI50 LLM Backend Service
Documentation=https://github.com/RaffaeleSpezia/local-llm-inference-lab
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=lele2
Group=lele2
WorkingDirectory=/mnt/raid0/services/mi50_ollama_like

# Environment
Environment="HIP_VISIBLE_DEVICES=0"
Environment="PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.95"
Environment="HF_HOME=/mnt/raid0/hf_cache"
Environment="TRANSFORMERS_CACHE=/mnt/raid0/hf_cache/transformers"
Environment="TORCH_HOME=/mnt/raid0/torch_cache"
Environment="OLLAMA_FAKE_LOGDIR=/dev/shm/mi50_ollama_logs"
Environment="OLLAMA_FAKE_DEFAULT_MODEL=/mnt/raid0/qwen2.5-coder-7b-instruct"
Environment="OLLAMA_FAKE_DEFAULT_MAX_NEW_TOKENS=128"
Environment="OLLAMA_FAKE_DEFAULT_TEMPERATURE=0.0"
Environment="OLLAMA_FAKE_ATTN_IMPL=sdpa"

# Execution
ExecStartPre=/bin/mkdir -p /dev/shm/mi50_ollama_logs
ExecStart=/mnt/raid0/shared_envs/venv-rocm311/bin/python app.py /mnt/raid0/qwen2.5-coder-7b-instruct --host 0.0.0.0 --port 11534 --log-level info

# Restart policy
Restart=on-failure
RestartSec=10s

# Resource limits (optional)
MemoryMax=64G
TasksMax=4096

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=mi50-backend

[Install]
WantedBy=multi-user.target
```

### Installazione

```bash
# Copia service file
sudo cp ~/mi50_stack/systemd/mi50-backend.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable (auto-start al boot)
sudo systemctl enable mi50-backend.service

# Start
sudo systemctl start mi50-backend.service

# Check status
sudo systemctl status mi50-backend.service
```

### Logs

```bash
# Tail logs
journalctl -u mi50-backend.service -f

# Logs last 100 lines
journalctl -u mi50-backend.service -n 100

# Logs since boot
journalctl -u mi50-backend.service -b
```

---

## Chat UI Service

### Service File

**Path:** `/etc/systemd/system/mi50-chat-ui.service`

```ini
[Unit]
Description=MI50 Chat UI Service
Documentation=https://github.com/RaffaeleSpezia/local-llm-inference-lab
After=network-online.target mi50-backend.service
Wants=network-online.target
Requires=mi50-backend.service

[Service]
Type=simple
User=lele2
Group=lele2
WorkingDirectory=/home/lele2/mi50_stack/mi50_chat_ui

# Environment
Environment="MI50_SERVER_URL=http://127.0.0.1:11534"
Environment="CHAT_UI_DEFAULT_MODEL=/mnt/raid0/qwen2.5-coder-7b-instruct"

# Execution
ExecStart=/mnt/raid0/shared_envs/venv-rocm311/bin/uvicorn app.main:app --host 0.0.0.0 --port 12000 --log-level info

# Restart policy
Restart=on-failure
RestartSec=5s

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=mi50-chat-ui

[Install]
WantedBy=multi-user.target
```

**Nota:** `Requires=mi50-backend.service` assicura che backend sia avviato prima.

### Installazione

```bash
sudo cp ~/mi50_stack/systemd/mi50-chat-ui.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable mi50-chat-ui.service
sudo systemctl start mi50-chat-ui.service
```

---

## Dashboard Service

### Service File

**Path:** `/etc/systemd/system/mi50-dashboard.service`

```ini
[Unit]
Description=MI50 Dashboard Service
After=network-online.target mi50-backend.service
Wants=network-online.target

[Service]
Type=simple
User=lele2
WorkingDirectory=/home/lele2/mi50_stack/mi50_dashboard

Environment="BACKEND_URL=http://127.0.0.1:11534"

ExecStart=/mnt/raid0/shared_envs/venv-rocm311/bin/python dashboard.py --port 13000

Restart=on-failure
RestartSec=5s

StandardOutput=journal
StandardError=journal
SyslogIdentifier=mi50-dashboard

[Install]
WantedBy=multi-user.target
```

### Installazione

```bash
sudo cp ~/mi50_stack/systemd/mi50-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable mi50-dashboard.service
sudo systemctl start mi50-dashboard.service
```

---

## Multi-Service Management

### Start All

```bash
sudo systemctl start mi50-backend.service mi50-chat-ui.service mi50-dashboard.service
```

### Stop All

```bash
sudo systemctl stop mi50-backend.service mi50-chat-ui.service mi50-dashboard.service
```

### Restart All

```bash
sudo systemctl restart mi50-backend.service mi50-chat-ui.service mi50-dashboard.service
```

### Status All

```bash
sudo systemctl status mi50-backend.service mi50-chat-ui.service mi50-dashboard.service
```

### Enable All (Auto-Start)

```bash
sudo systemctl enable mi50-backend.service mi50-chat-ui.service mi50-dashboard.service
```

### Logs All

```bash
journalctl -u mi50-backend.service -u mi50-chat-ui.service -u mi50-dashboard.service -f
```

---

## Target Service (Optional)

**Semplifica management:** un singolo comando per tutti i servizi.

### mi50-stack.target

**Path:** `/etc/systemd/system/mi50-stack.target`

```ini
[Unit]
Description=MI50 Stack Services Target
Requires=mi50-backend.service mi50-chat-ui.service mi50-dashboard.service
After=mi50-backend.service mi50-chat-ui.service mi50-dashboard.service

[Install]
WantedBy=multi-user.target
```

### Installazione

```bash
sudo cp ~/mi50_stack/systemd/mi50-stack.target /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable mi50-stack.target
```

### Utilizzo

```bash
# Start all
sudo systemctl start mi50-stack.target

# Stop all
sudo systemctl stop mi50-stack.target

# Status
sudo systemctl status mi50-stack.target
```

---

## Troubleshooting

### Service non parte

```bash
# Check status
sudo systemctl status mi50-backend.service

# Logs dettagliati
journalctl -u mi50-backend.service -n 100 --no-pager

# Verifica permessi
ls -la /mnt/raid0/services/mi50_ollama_like/app.py

# Test manuale
cd /mnt/raid0/services/mi50_ollama_like
source /mnt/raid0/shared_envs/venv-rocm311/bin/activate
python app.py /mnt/raid0/qwen2.5-coder-7b-instruct
```

### Permission denied

```bash
# Fix ownership
sudo chown -R lele2:lele2 /home/lele2/mi50_stack
sudo chown -R lele2:lele2 /mnt/raid0/services
```

### Port già in uso

```bash
# Trova processo
sudo lsof -iTCP:11534 -sTCP:LISTEN

# Killa processo
sudo kill <PID>
```

---

## Next Steps

**→ [07-troubleshooting.md](./07-troubleshooting.md)** - Risoluzione problemi comuni

---

**Licenza:** CC BY-NC-SA 4.0
**Autore:** Raffaele Spezia
**Repository:** https://github.com/RaffaeleSpezia/local-llm-inference-lab
