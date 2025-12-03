# Hardware Setup - MI50 Inference Server

**Versione:** 1.0
**Data:** 3 Dicembre 2025
**Autore:** Raffaele Spezia

---

## Indice

1. [Overview Hardware](#overview-hardware)
2. [GPU AMD MI50](#gpu-amd-mi50)
3. [GPU NVIDIA Tesla M40](#gpu-nvidia-tesla-m40)
4. [Configurazione Network](#configurazione-network)
5. [Storage e RAID](#storage-e-raid)
6. [Verifica Hardware](#verifica-hardware)
7. [Driver e Firmware](#driver-e-firmware)

---

## Overview Hardware

Il sistema MI50 Stack utilizza **due GPU dedicate** su un singolo server per ottimizzare performance e separazione dei carichi di lavoro.

### Architettura Dual-GPU

| Componente | Specifiche | Ruolo | Porta Servizi |
|------------|------------|-------|---------------|
| **Server** | 192.168.1.155 | Host principale | - |
| **GPU 1** | AMD MI50 32GB | LLM Generation | 11534, 11535 |
| **GPU 2** | NVIDIA Tesla M40 12GB | RAG Embeddings (CPU) | 11600 |
| **RAM** | 168GB | Model Loading Buffer | - |
| **Storage** | RAID0 /mnt/raid0 | Models + Cache | - |
| **Network** | Gigabit LAN 192.168.1.x | API Services | - |

### Vantaggi Separazione GPU

**Perché due GPU invece di una?**

1. **Zero competizione VRAM**: MI50 dedica tutti i 32GB al modello LLM, M40 usa CPU per embeddings
2. **Fault tolerance**: Se un servizio crasha, l'altro continua
3. **Scalabilità indipendente**: Posso ottimizzare ogni GPU per il suo task
4. **Costo-efficacia**: M40 vecchia ma perfetta per task leggeri (attualmente usa CPU)

---

## GPU AMD MI50

### Specifiche Tecniche

| Parametro | Valore |
|-----------|--------|
| **Architettura** | GCN 5.0 (Vega 20) |
| **VRAM** | 32 GB HBM2 |
| **Memoria Bandwidth** | 1024 GB/s |
| **Compute Units** | 60 CU |
| **Stream Processors** | 3840 SP |
| **FP32 Performance** | 13.3 TFLOPS |
| **FP16 Performance** | 26.5 TFLOPS |
| **TDP** | 300W |
| **Compute Capability** | ROCm 9.0 |

### Utilizzo nel Sistema

**Task principale:** Generazione testo con LLM di grandi dimensioni

**Modelli supportati:**
- Qwen 2.5 Coder 7B (~14GB VRAM)
- Qwen 2.5 Coder 14B (~28GB VRAM)
- Gemma 2 9B IT (~18GB VRAM)
- DeepSeek Coder 6.7B (~13GB VRAM)
- Gemma3 4B IT (~8GB VRAM)

**Performance tipiche:**
- VRAM idle: 45-50% (14-16GB)
- VRAM inferenza: 60-75% (20-24GB)
- Throughput: 50-100 tokens/sec (dipende da modello)
- Latency: 2-5s per prompt brevi (<100 token)
- Temperatura: 25-30°C idle, 40-50°C carico

### Ottimizzazione VRAM

**Problema risolto (Ottobre 2025):**

Prima dell'ottimizzazione, il sistema usava **90% VRAM idle** (28GB su 32GB) a causa di double allocation durante caricamento modello:

```
❌ PRIMA: low_cpu_mem_usage=True + device_map="auto"
→ PyTorch alloca buffer VRAM durante loading
→ Modello 7B: 14GB reali + 14GB buffer = 28GB sprecati
→ Solo 4GB liberi per KV cache → swap continuo → lentezza
```

**Soluzione implementata:**

Sfruttare i **168GB RAM** disponibili come buffer intermedio:

```python
# In model_manager.py
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16  # NO device_map, NO low_cpu_mem_usage
)
model = model.to("cuda:0")      # Transfer diretto RAM → VRAM
torch.cuda.empty_cache()        # Cleanup buffer
```

**Risultato:**
- ✅ VRAM idle: 50% (16GB invece di 28GB)
- ✅ 16GB liberi per KV cache
- ✅ Velocità 10x: da 5-10 tok/s a 50-100 tok/s
- ✅ Latenza ridotta: da 20-30s a 2-5s

Per dettagli completi: `RAM_VRAM_OPTIMIZATION.md` (nel backend)

### Profilo Performance GPU

**Impostazione critica:**

```bash
# SEMPRE dopo boot del server
sudo rocm-smi --setperflevel high
```

**Perché?**
- Default `auto` non scala oltre 925MHz
- `high` forza clock massimi (anche se su questo server rimane 925MHz, abilita allocator ottimale)
- Migliora utilizzo GPU del 20-30%

**Verifica:**

```bash
rocm-smi
# Output atteso:
# GPU  Temp  AvgPwr  SCLK     MCLK     Fan     Perf  PwrCap  VRAM%
#   0  30°C  120W    925MHz   1000MHz  30%     high  300W    50%
```

### Configurazione ROCm

**Versione:** ROCm 6.2.4 (con PyTorch 2.5.1+rocm6.2)

**Variabili ambiente obbligatorie:**

```bash
# In ~/.bashrc o start script
export HIP_VISIBLE_DEVICES=0          # Usa solo MI50 (indice 0)
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.95"
```

**Cosa fanno:**
- `HIP_VISIBLE_DEVICES=0`: Nasconde M40, evita conflitti
- `PYTORCH_HIP_ALLOC_CONF`: Allocator aggressivo per liberare VRAM
  - `max_split_size_mb`: Limite frammenti memoria
  - `garbage_collection_threshold`: Soglia pulizia (95%)

---

## GPU NVIDIA Tesla M40

### Specifiche Tecniche

| Parametro | Valore |
|-----------|--------|
| **Architettura** | Maxwell (GM200) |
| **VRAM** | 12 GB GDDR5 |
| **Memoria Bandwidth** | 288 GB/s |
| **CUDA Cores** | 3072 |
| **Compute Capability** | 5.2 |
| **TDP** | 250W |

### Utilizzo nel Sistema

**Task principale:** RAG Embeddings generation

**Nota importante:** Attualmente il sistema M40 usa **CPU per embeddings**, non GPU:
- ONNX Runtime (CPU): più stabile
- Embedding model: `intfloat/multilingual-e5-small` (384 dimensioni)
- VRAM M40: 0 MB utilizzata
- Temperatura: ~11°C (idle completo)
- Power: ~16W

**Performance embeddings (CPU):**
- Latency: 3-8ms per embedding
- Throughput: ~125 embeddings/sec
- Accuracy: 85%+ similarity search

**Perché CPU invece di GPU?**
1. Embeddings sono task leggero (384D, non 4096D)
2. ONNX CPU è più stabile di CUDA per questo workload
3. Libera M40 per esperimenti futuri
4. Temperatura/consumi ridotti (11°C vs 50-60°C)

### Possibili Utilizzi Futuri M40

La M40 è disponibile per:
- Fine-tuning modelli piccoli (fino a 7B con quantizzazione)
- Embeddings GPU-accelerati (se serve più throughput)
- Secondo modello LLM leggero (Gemma3 4B, Phi-3)
- Esperimenti con vLLM

---

## Configurazione Network

### Dettagli Server

**Hostname:** `lele2@192.168.1.155`
**Password:** `pippopippo`
**Rete:** LAN privata 192.168.1.0/24
**Accesso:** SSH (porta 22)

### Porte Servizi

| Porta | Servizio | Descrizione | GPU |
|-------|----------|-------------|-----|
| **11534** | Backend MI50 | LLM generation API | MI50 |
| **11535** | RAG Proxy | Proxy intelligente con RAG automatico | - |
| **11600** | RAG Server M40 | Embeddings + ChromaDB | M40 (CPU) |
| **12000** | Chat UI | Interfaccia web chat | - |
| **13000** | Dashboard | Monitoring + Admin UI | - |
| **14000** | ESP32 Generator | Code generator per ESP32 | - |
| **18500** | Evolve | Fine-tuning UI | - |

### Firewall

**Regole necessarie (se attivo):**

```bash
# Apri porte servizi
sudo ufw allow 11534/tcp comment 'Backend MI50'
sudo ufw allow 11535/tcp comment 'RAG Proxy'
sudo ufw allow 11600/tcp comment 'RAG M40'
sudo ufw allow 12000/tcp comment 'Chat UI'
sudo ufw allow 13000/tcp comment 'Dashboard'
sudo ufw allow 14000/tcp comment 'ESP32 Generator'
sudo ufw allow 18500/tcp comment 'Evolve'

# Verifica
sudo ufw status
```

### Test Connettività

**Da PC locale (lele):**

```bash
# SSH
ssh lele2@192.168.1.155

# Test backend MI50
curl http://192.168.1.155:11534/api/version

# Test RAG M40
curl http://192.168.1.155:11600/health

# Test Chat UI
firefox http://192.168.1.155:12000
```

---

## Storage e RAID

### Configurazione RAID0

**Mount point:** `/mnt/raid0`
**Tipo:** RAID0 (stripe)
**Scopo:** Velocità massima per model loading

### Directory Structure

```
/mnt/raid0/
├── models/                    # Modelli LLM (100-200GB totali)
│   ├── qwen2.5-coder-7b-instruct/     (~14GB)
│   ├── qwen2.5-coder-14b-instruct/    (~28GB)
│   ├── gemma-2-9b-it/                 (~18GB)
│   ├── deepseek-coder-6.7b-instruct/  (~13GB)
│   └── gemma3-4b-it/                  (~8GB)
│
├── hf_cache/                  # HuggingFace cache
│   └── transformers/
│
├── torch_cache/               # PyTorch cache
│
├── shared_envs/               # Virtual environments
│   └── venv-rocm311/
│
└── services/                  # Service directories
    ├── mi50_ollama_like/      # Backend MI50
    ├── rag_server_m40/        # RAG M40
    └── ...
```

### Cache Configuration

**Variabili ambiente per evitare duplicazione cache:**

```bash
export HF_HOME=/mnt/raid0/hf_cache
export TRANSFORMERS_CACHE=/mnt/raid0/hf_cache/transformers
export TORCH_HOME=/mnt/raid0/torch_cache
```

**Perché importante:**
- Senza queste variabili, ogni venv scarica copie duplicate dei modelli
- Con path condivisi: 1 copia = 14GB, 5 copie = 70GB sprecati
- RAID velocizza caricamento modelli (500MB/s vs 150MB/s su HDD singolo)

### Spazio Disco

**Check utilizzo:**

```bash
df -h /mnt/raid0

# Output tipico:
# Filesystem      Size  Used Avail Use% Mounted on
# /dev/md0        2.0T  350G  1.6T  18% /mnt/raid0
```

**Cleanup periodico:**

```bash
# Rimuovi cache HuggingFace vecchia
rm -rf /mnt/raid0/hf_cache/transformers/models--*/.locks

# Rimuovi log vecchi (opzionale)
find /mnt/raid0/services/*/logs -name "*.log" -mtime +30 -delete
```

---

## Verifica Hardware

### Check GPU Status

**AMD MI50:**

```bash
rocm-smi

# Output atteso:
# ========================ROCm System Management Interface========================
# ================================================================================
# GPU  Temp   AvgPwr  SCLK     MCLK     Fan     Perf  PwrCap  VRAM%  GPU%
#   0  30°C   120W    925MHz   1000MHz  Auto    high  300W    50%    0%
# ================================================================================
```

**NVIDIA M40:**

```bash
nvidia-smi

# Output atteso (idle, no GPU usage):
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.XX       Driver Version: 525.XX       CUDA Version: 12.0    |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        TCC/WDDM  | Bus-Id        Disp.A | Volatile Uncorr. ECC |
# |   0  Tesla M40          Off  | 00000000:03:00.0 Off |                    0 |
# | N/A   11C    P8    16W / 250W |      0MiB / 12288MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

### Check PyTorch/ROCm

**Test MI50 visibility:**

```bash
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'Device Name: {torch.cuda.get_device_name(0)}')"

# Output atteso:
# CUDA Available: True
# Device Count: 1
# Device Name: AMD Radeon Instinct MI50
```

### Check RAM

```bash
free -h

# Output atteso:
#               total        used        free      shared  buff/cache   available
# Mem:           168G        25G        120G        1.2G        22G        140G
# Swap:          16G         0B         16G
```

**Nota:** 168GB RAM è cruciale per la strategia di ottimizzazione VRAM.

### Health Check Script

Crea `/home/lele2/check_hw.sh`:

```bash
#!/bin/bash

echo "=== GPU MI50 ==="
rocm-smi | grep -A1 "GPU  Temp"

echo -e "\n=== GPU M40 ==="
nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv

echo -e "\n=== PyTorch CUDA ==="
python3 -c "import torch; print(f'Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo -e "\n=== RAM ==="
free -h | grep Mem

echo -e "\n=== Disk /mnt/raid0 ==="
df -h /mnt/raid0 | grep -v Filesystem

echo -e "\n=== Services Status ==="
ss -tulnp | grep -E ":(11534|11535|11600|12000|13000)" | awk '{print $5}' | cut -d: -f2 | sort -u
```

Run:
```bash
chmod +x ~/check_hw.sh
./check_hw.sh
```

---

## Driver e Firmware

### ROCm Installation

**Versione installata:** ROCm 6.2.4

**Path installazione:**
- `/opt/rocm-6.2.4/`
- Symlink: `/opt/rocm` → `/opt/rocm-6.2.4`

**Verifica versione:**

```bash
/opt/rocm/bin/rocm-smi --version
# Output: ROCm 6.2.4

/opt/rocm/bin/rocminfo | grep "Name:" | head -1
# Output: Name: gfx906 (MI50)
```

**PATH configuration:**

```bash
# In ~/.bashrc
export PATH=/opt/rocm/bin:$PATH
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### PyTorch ROCm Build

**Versione:** PyTorch 2.5.1+rocm6.2

**Check installazione:**

```bash
python3 -c "import torch; print(torch.__version__)"
# Output: 2.5.1+rocm6.2

python3 -c "import torch; print(torch.version.hip)"
# Output: 6.2.41134-ef8d0e878
```

**Reinstall se necessario:**

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/rocm6.2
```

### NVIDIA Drivers (M40)

**Versione:** NVIDIA Driver 525.x + CUDA 12.0

**Verifica:**

```bash
nvidia-smi
# Header mostra: Driver Version: 525.XX  CUDA Version: 12.0
```

**Nota:** Non serve CUDA Toolkit completo, solo driver. Il servizio RAG usa CPU.

---

## Troubleshooting Hardware

### MI50: "No GPU Available"

**Sintomo:**
```python
torch.cuda.is_available() → False
```

**Cause:**
1. ROCm non installato/configurato
2. `HIP_VISIBLE_DEVICES` sbagliato
3. Driver non caricato

**Fix:**

```bash
# Check driver
lsmod | grep amdgpu
# Se vuoto → driver non caricato

# Reload driver (richiede sudo)
sudo modprobe amdgpu

# Check device
ls -la /dev/kfd
# Deve esistere

# Verifica gruppi utente
groups lele2
# Deve contenere "video" e "render"
```

### MI50: VRAM al 90% Idle

**Sintomo:**
```bash
rocm-smi | grep VRAM%
# Output: VRAM% = 90%+ con modello appena caricato
```

**Causa:** Double allocation bug (risolto)

**Fix:** Verifica che `model_manager.py` NON contenga:
```python
# ❌ Questi causano il problema
low_cpu_mem_usage=True
device_map="auto"
```

Deve contenere:
```python
# ✅ Versione corretta
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)
model = model.to("cuda:0")
torch.cuda.empty_cache()
```

### M40: GPU Overheating

**Sintomo:**
```bash
nvidia-smi
# Temp > 80°C
```

**Cause:**
- GPU erroneamente utilizzata per task (dovrebbe essere CPU)
- Fan failure

**Fix:**

```bash
# Check utilizzo GPU
nvidia-smi dmon -s pucvmet
# GPU%  MEM% dovrebbero essere 0%

# Se GPU% > 0 → verifica quale processo
nvidia-smi pmon
```

### RAM Insufficiente

**Sintomo:**
- Backend crash durante caricamento modello
- `OOMKilled` nei log systemd

**Cause:**
- Modello troppo grande (es. 14B con 32GB RAM sarebbe tight)
- Altri processi consumano RAM

**Fix:**

```bash
# Check RAM disponibile
free -h

# Libera cache
sudo sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Identifica processi RAM-heavy
ps aux --sort=-%mem | head -10
```

---

## Next Steps

Con l'hardware configurato e verificato, procedi a:

**→ [02-software-stack.md](./02-software-stack.md)** - Installazione ROCm, PyTorch, dipendenze Python

---

**Licenza:** CC BY-NC-SA 4.0
**Autore:** Raffaele Spezia
**Repository:** https://github.com/RaffaeleSpezia/local-llm-inference-lab
