# Troubleshooting - Risoluzione Problemi Comuni

**Versione:** 1.0
**Data:** 3 Dicembre 2025
**Autore:** Raffaele Spezia

---

## Indice

1. [Hardware Issues](#hardware-issues)
2. [Backend Issues](#backend-issues)
3. [Performance Issues](#performance-issues)
4. [Network Issues](#network-issues)
5. [Service Management](#service-management)
6. [Debug Tools](#debug-tools)

---

## Hardware Issues

### MI50: GPU Not Detected

**Sintomo:**
```bash
rocm-smi
# Error: No GPU detected
```

**Diagnosi:**
```bash
# 1. Check PCI device
lspci | grep -i amd
# Deve mostrare: Vega 20 [Radeon Instinct MI50]

# 2. Check driver loaded
lsmod | grep amdgpu
# Se vuoto → driver non caricato

# 3. Check device node
ls -la /dev/kfd /dev/dri/renderD*
# Devono esistere
```

**Fix:**
```bash
# Reload driver
sudo modprobe amdgpu

# Se fallisce → reinstall ROCm
sudo amdgpu-install --usecase=rocm

# Reboot (last resort)
sudo reboot
```

**Verifiche post-fix:**
```bash
rocm-smi
rocminfo | grep "Name:" | head -1  # Deve mostrare gfx906
python3 -c "import torch; print(torch.cuda.is_available())"  # True
```

---

### PyTorch: "CUDA Not Available"

**Sintomo:**
```python
import torch
print(torch.cuda.is_available())  # False
```

**Cause possibili:**
1. ROCm non installato
2. PyTorch CPU-only (non ROCm build)
3. `HIP_VISIBLE_DEVICES` misconfigured
4. User non in gruppo video/render

**Diagnosi:**
```bash
# 1. Check PyTorch version
python3 -c "import torch; print(torch.__version__)"
# Deve contenere "+rocm" (es: 2.5.1+rocm6.2)

# 2. Check HIP
python3 -c "import torch; print(torch.version.hip)"
# Non deve essere None

# 3. Check groups
groups
# Deve contenere: video render

# 4. Check HIP_VISIBLE_DEVICES
echo $HIP_VISIBLE_DEVICES  # Deve essere 0 o vuoto (non 1, non -1)
```

**Fix:**
```bash
# Fix 1: Reinstall PyTorch ROCm
pip uninstall torch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2

# Fix 2: Add user to groups
sudo usermod -a -G video,render $USER
# Logout e login

# Fix 3: Correct HIP_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=0
```

---

### VRAM al 90% Idle (BUG RISOLTO)

**Sintomo:**
```bash
rocm-smi
# VRAM% = 90%+ subito dopo caricamento modello
```

**Root cause:** Double allocation durante model loading (bug risolto Ottobre 2025).

**Verifica fix applicato:**
```bash
cd ~/mi50_stack/mi50_come_ollama
grep -n "low_cpu_mem_usage" model_manager.py
# NON deve apparire

grep -n "device_map" model_manager.py
# NON deve apparire nel load_model()

grep -n "torch.cuda.empty_cache()" model_manager.py
# DEVE apparire 3 volte (line ~148, ~186, ~195)
```

**Se bug presente:**
```bash
# Backup current
cp model_manager.py model_manager.py.backup_$(date +%Y%m%d_%H%M%S)

# Apply fix: rimuovi low_cpu_mem_usage e device_map da load_model()
# Aggiungi torch.cuda.empty_cache() dopo model.to("cuda:0")
```

**Test fix:**
```bash
# Restart backend
pkill pt_main_t
./start.sh

# Check VRAM dopo 2 minuti
rocm-smi | grep VRAM%
# Atteso: 45-50% (non 90%)

# Debug endpoint
curl http://127.0.0.1:11534/debug/memory | jq
# waste_gb dovrebbe essere < 2GB
```

---

## Backend Issues

### Backend non parte

**Sintomo:**
```bash
./start.sh
# Error o crash immediato
```

**Diagnosi step-by-step:**

**1. Check dependencies:**
```bash
source /mnt/raid0/shared_envs/venv-rocm311/bin/activate
python3 -c "import torch, transformers, fastapi"
# Se ImportError → reinstall
pip install -r requirements.txt
```

**2. Check modello esiste:**
```bash
ls -la /mnt/raid0/qwen2.5-coder-7b-instruct/
# Deve contenere: config.json, *.safetensors, tokenizer*
```

**3. Check VRAM libera:**
```bash
rocm-smi | grep VRAM%
# Se > 80% → kill processi GPU
pkill pt_main_t
```

**4. Check porta libera:**
```bash
lsof -iTCP:11534 -sTCP:LISTEN
# Se occupata → kill processo
kill <PID>
```

**5. Test minimale:**
```bash
cd ~/mi50_stack/mi50_come_ollama
source /mnt/raid0/shared_envs/venv-rocm311/bin/activate
export HIP_VISIBLE_DEVICES=0
python3 << EOF
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/raid0/qwen2.5-coder-7b-instruct")
print(f"Tokenizer OK: {len(tokenizer)} tokens")
EOF
```

---

### Model Loading Timeout

**Sintomo:**
```bash
curl http://127.0.0.1:11534/api/version
# Timeout dopo 30s
```

**Causa:** Model loading richiede 1-2 minuti (normale per modelli 7B+).

**Diagnosi:**
```bash
# Check log
tail -f /dev/shm/mi50_ollama_logs/mi50_ollama.log
# Atteso: "Loading model..." → "Model loaded successfully" (1-2 min)

# Check processo attivo
ps aux | grep "python.*app.py"
# Deve mostrare processo running

# Check VRAM crescente
watch -n 2 rocm-smi
# VRAM% dovrebbe crescere gradualmente durante loading
```

**Fix:**
- **Aspetta 2 minuti** prima di considerare errore
- Se dopo 5 minuti ancora timeout → check log errori

---

### Generation Lentissima (< 10 tok/s)

**Sintomo:**
```bash
curl -X POST http://127.0.0.1:11534/api/generate \
  -d '{"prompt":"Hello"}' -H 'Content-Type: application/json'
# Risponde ma molto lento (< 10 tokens/sec)
```

**Cause:**
1. VRAM satura (> 90%) → swap
2. GPU performance level = auto (non high)
3. Modello troppo grande (14B su VRAM limitata)
4. Context troppo lungo

**Diagnosi:**
```bash
# 1. Check VRAM
curl http://127.0.0.1:11534/debug/memory | jq
# Se allocated > 28GB → problema

rocm-smi | grep -E "VRAM%|Perf"
# Perf deve essere "high"

# 2. Check tokens/sec in log
tail -100 /dev/shm/mi50_ollama_logs/mi50_ollama.log | grep "tok/s"
```

**Fix:**
```bash
# Fix 1: Set GPU perf high
sudo rocm-smi --setperflevel high

# Fix 2: Reload backend (libera VRAM)
pkill pt_main_t
sleep 5
./start.sh

# Fix 3: Usa modello più piccolo
# In start.sh, cambia:
# MODEL="/mnt/raid0/qwen2.5-coder-7b-instruct"  # Invece di 14B
```

---

### Tool Calling Not Working

**Sintomo:**
```bash
curl -X POST http://127.0.0.1:11534/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[...], "tools":[...]}'
# Response: no tool_calls, solo text
```

**Cause:**
1. Modello non supporta tool calling (solo Qwen 2.5+ supporta)
2. Tool definition malformata
3. Prompt non include tool instructions

**Diagnosi:**
```bash
# Check modello
curl http://127.0.0.1:11534/api/tags | jq -r '.models[0].name'
# Deve contenere "qwen2.5" o "qwen3"

# Check log tool parsing
tail -100 /dev/shm/mi50_ollama_logs/mi50_ollama.log | grep -i tool
```

**Fix:**
```bash
# Fix 1: Usa Qwen 2.5+
# Qwen 2.5, DeepSeek v3+ supportano tool calling
# Gemma, Llama base NON supportano

# Fix 2: Verifica tool schema OpenAI-compliant
# Deve avere: type, function.name, function.parameters
```

---

## Performance Issues

### Inferenza Lenta su Modelli 14B+

**Sintomo:** Qwen 14B genera < 30 tok/s (atteso 50-70).

**Cause:**
- VRAM insufficiente → parte del modello in RAM → swap
- KV cache limitata

**Diagnosi:**
```bash
curl http://127.0.0.1:11534/debug/memory | jq
# Se VRAM allocated > 30GB → problema

# 14B FP16 richiede ~28GB VRAM
# MI50 ha 32GB → solo 4GB liberi per KV cache
```

**Fix:**
```bash
# Fix 1: Usa modello quantizzato (GPTQ Int4)
# Qwen 14B GPTQ-Int4 → 14GB VRAM invece di 28GB

# Fix 2: Ridurre max_new_tokens
# Meno token = meno KV cache richiesta

# Fix 3: Usa modello 7B invece di 14B
# Trade-off: qualità vs velocità
```

---

### Context Overflow

**Sintomo:**
```bash
curl -X POST http://127.0.0.1:11534/api/generate \
  -d '{"prompt":"<very long prompt>"}}'
# Error: "Prompt too long, truncated to 4096 tokens"
```

**Causa:** `OLLAMA_FAKE_MAX_PROMPT_TOKENS=4096` (default).

**Fix:**
```bash
# Aumenta limite
export OLLAMA_FAKE_MAX_PROMPT_TOKENS=8192

# Restart backend
pkill pt_main_t
./start.sh

# Oppure usa modello con context più grande
# Qwen 2.5 → 32768 tokens context
```

---

## Network Issues

### Connection Refused

**Sintomo:**
```bash
curl http://192.168.1.155:11534/api/version
# curl: (7) Failed to connect: Connection refused
```

**Diagnosi:**
```bash
# 1. Check servizio running
ps aux | grep "python.*app.py"

# 2. Check porta listening
ss -tulnp | grep 11534

# 3. Check firewall
sudo ufw status | grep 11534
```

**Fix:**
```bash
# Fix 1: Start backend
cd ~/mi50_stack/mi50_come_ollama
./start.sh

# Fix 2: Apri porta firewall
sudo ufw allow 11534/tcp

# Fix 3: Check binding (deve essere 0.0.0.0, non 127.0.0.1)
# In app.py:
# app.run(host="0.0.0.0", port=11534)  # ✓
# app.run(host="127.0.0.1", port=11534)  # ✗ (solo locale)
```

---

### Timeout su Generazioni Lunghe

**Sintomo:**
```bash
curl --max-time 60 -X POST http://192.168.1.155:11534/api/generate \
  -d '{"prompt":"Write long essay", "options":{"max_new_tokens":2048}}'
# Error: Operation timed out after 60000 milliseconds
```

**Causa:** Generation > timeout client.

**Fix:**
```bash
# Fix client: aumenta timeout
curl --max-time 300 ...  # 5 minuti

# O usa streaming
curl -N -X POST ... -d '{"prompt":"...", "stream":true}'
# Streaming evita timeout (response continua)
```

---

## Service Management

### Systemd Service Fails

**Sintomo:**
```bash
sudo systemctl start mi50-backend.service
# Job for mi50-backend.service failed
```

**Diagnosi:**
```bash
# 1. Check status dettagliato
sudo systemctl status mi50-backend.service -l

# 2. Logs ultimi 100 messaggi
journalctl -u mi50-backend.service -n 100 --no-pager

# 3. Check permission file service
ls -la /etc/systemd/system/mi50-backend.service
# Deve essere -rw-r--r-- root:root

# 4. Test manuale con stesso comando ExecStart
cd /mnt/raid0/services/mi50_ollama_like
source /mnt/raid0/shared_envs/venv-rocm311/bin/activate
export HIP_VISIBLE_DEVICES=0
python app.py /mnt/raid0/qwen2.5-coder-7b-instruct --port 11534
```

**Fix:**
```bash
# Fix 1: Correggi permessi
sudo chown lele2:lele2 /home/lele2/mi50_stack

# Fix 2: Fix working directory
# In service file: WorkingDirectory deve esistere

# Fix 3: Reload systemd dopo modifiche
sudo systemctl daemon-reload
sudo systemctl start mi50-backend.service
```

---

## Debug Tools

### Log Monitoring

**Tail log real-time:**
```bash
# Backend
tail -f /dev/shm/mi50_ollama_logs/mi50_ollama.log

# Systemd
journalctl -u mi50-backend.service -f

# Tutti i servizi
journalctl -u mi50-backend.service -u mi50-chat-ui.service -u mi50-dashboard.service -f
```

**Search log errori:**
```bash
# Errori backend last hour
journalctl -u mi50-backend.service --since "1 hour ago" | grep -i error

# Pattern OOM
grep -i "out of memory" /dev/shm/mi50_ollama_logs/mi50_ollama.log
```

---

### VRAM Monitoring

**Watch VRAM real-time:**
```bash
watch -n 1 'rocm-smi | grep -A1 "GPU.*Temp"'
```

**Plot VRAM history:**
```bash
while true; do
  vram=$(rocm-smi | awk '/VRAM%/{print $8}' | tr -d '%')
  echo "$(date +%H:%M:%S),$vram" >> vram_log.csv
  sleep 2
done
```

---

### Network Debug

**Test endpoints:**
```bash
# Health check
curl -s http://192.168.1.155:11534/api/version | jq

# Memory debug
curl -s http://192.168.1.155:11534/debug/memory | jq

# Quick generation test
time curl -s -X POST http://192.168.1.155:11534/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Say hello","options":{"max_new_tokens":10}}' | jq -r '.response'
```

**Check latency:**
```bash
# Ping server
ping -c 5 192.168.1.155

# HTTP latency
time curl -s http://192.168.1.155:11534/api/version > /dev/null
```

---

## Emergency Recovery

### Complete System Reset

**Quando tutto fallisce:**

```bash
# 1. Kill all services
pkill -f "python.*app.py"
pkill -f uvicorn

# 2. Clear VRAM
# (Automatic quando processi killati)

# 3. Check hardware
rocm-smi
nvidia-smi
free -h

# 4. Restart from scratch
cd ~/mi50_stack
./start_all_services.sh

# 5. Monitor startup
tail -f /tmp/mi50_services_logs/*.log

# 6. Test
curl http://192.168.1.155:11534/api/version
firefox http://192.168.1.155:12000
```

---

## Next Steps

**→ [08-performance-tuning.md](./08-performance-tuning.md)** - Ottimizzazione performance avanzata

---

**Licenza:** CC BY-NC-SA 4.0
**Autore:** Raffaele Spezia
**Repository:** https://github.com/RaffaeleSpezia/local-llm-inference-lab
