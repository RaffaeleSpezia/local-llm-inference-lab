# Software Stack - MI50 Inference Server

**Versione:** 1.0
**Data:** 3 Dicembre 2025
**Autore:** Raffaele Spezia

---

## Indice

1. [Overview Stack](#overview-stack)
2. [Sistema Operativo](#sistema-operativo)
3. [ROCm Installation](#rocm-installation)
4. [PyTorch ROCm](#pytorch-rocm)
5. [Python Environment](#python-environment)
6. [Dipendenze Chiave](#dipendenze-chiave)
7. [Modelli LLM](#modelli-llm)
8. [Verifica Installazione](#verifica-installazione)

---

## Overview Stack

### Layer Software

```
┌─────────────────────────────────────────────────────┐
│         Applications Layer                          │
│  Chat UI, Dashboard, ESP32 Generator, RAG Proxy     │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│         Framework Layer                             │
│  FastAPI, Flask, Uvicorn, Streamlit                 │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│         AI/ML Layer                                 │
│  PyTorch 2.5.1+rocm6.2, Transformers 4.45,          │
│  ONNX Runtime 1.23, ChromaDB 1.3.5                  │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│         Compute Layer                               │
│  ROCm 6.2.4 (MI50), CUDA 12.0 (M40)                 │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│         Hardware Layer                              │
│  AMD MI50 32GB, NVIDIA Tesla M40 12GB, 168GB RAM    │
└─────────────────────────────────────────────────────┘
```

### Versioni Software

| Componente | Versione | Note |
|------------|----------|------|
| **OS** | Ubuntu 22.04 LTS | Kernel 6.5+ |
| **Python** | 3.12 | Backend MI50, Chat UI, Dashboard |
| **Python** | 3.11 | RAG M40 (ONNX compatibility) |
| **ROCm** | 6.2.4 | AMD GPU compute |
| **PyTorch** | 2.5.1+rocm6.2 | Deep learning MI50 |
| **ONNX Runtime** | 1.23 | Embeddings CPU M40 |
| **Transformers** | 4.45 | HuggingFace models |
| **FastAPI** | 0.122 | REST API framework |
| **ChromaDB** | 1.3.5 | Vector database |
| **CUDA** | 12.0 | NVIDIA drivers (M40) |

---

## Sistema Operativo

### Ubuntu 22.04 LTS

**Kernel:** Linux 6.5+
**Architettura:** x86_64

**Verifica:**

```bash
lsb_release -a
# Output:
# Distributor ID: Ubuntu
# Description:    Ubuntu 22.04.3 LTS
# Release:        22.04
# Codename:       jammy

uname -r
# Output: 6.5.0-XX-generic
```

### Pacchetti Sistema Richiesti

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  wget \
  curl \
  python3-pip \
  python3-dev \
  python3-venv \
  libssl-dev \
  libffi-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  tk-dev \
  libpq-dev
```

### System Libraries

```bash
# Per ONNX Runtime
sudo apt install -y libgomp1

# Per ChromaDB
sudo apt install -y libsqlite3-0

# Per compilazione estensioni Python
sudo apt install -y gcc g++ make
```

---

## ROCm Installation

### Requisiti Pre-Installazione

**Verifica supporto GPU:**

```bash
lspci | grep -i amd
# Output atteso:
# 0c:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Vega 20 [Radeon Instinct MI50]
```

### Installazione ROCm 6.2.4

**Step 1: Add AMD Repository**

```bash
# Download installer
wget https://repo.radeon.com/amdgpu-install/6.2.4/ubuntu/jammy/amdgpu-install_6.2.60204-1_all.deb

# Install package
sudo apt install ./amdgpu-install_6.2.60204-1_all.deb

# Update repos
sudo apt update
```

**Step 2: Install ROCm**

```bash
# Full ROCm stack per ML
sudo amdgpu-install --usecase=rocm --no-dkms

# O installazione completa (include dkms)
sudo amdgpu-install --usecase=rocm
```

**Step 3: Add User to Groups**

```bash
# Necessario per accesso GPU
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER

# Logout/login per applicare
```

**Step 4: Set Environment Variables**

Aggiungi a `~/.bashrc`:

```bash
# ROCm paths
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# HIP configuration
export HIP_VISIBLE_DEVICES=0  # Only MI50
export HSA_OVERRIDE_GFX_VERSION=9.0.6  # MI50 = gfx906
```

Apply:
```bash
source ~/.bashrc
```

### Verifica ROCm

```bash
# Check version
rocm-smi --version
# Output: ROCm 6.2.4

# Check GPU detection
rocm-smi
# Output: Should show MI50 stats

# Check ROCm info
rocminfo | grep "Name:" | head -1
# Output: Name: gfx906

# Check HIP
hipconfig
# Output: HIP version + paths
```

### Performance Level Configuration

**IMPORTANTE:** Dopo ogni reboot, eseguire:

```bash
sudo rocm-smi --setperflevel high
```

**Automazione via systemd:**

Crea `/etc/systemd/system/rocm-performance.service`:

```ini
[Unit]
Description=Set ROCm GPU Performance Level
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/opt/rocm/bin/rocm-smi --setperflevel high
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable rocm-performance.service
sudo systemctl start rocm-performance.service
```

---

## PyTorch ROCm

### Installazione PyTorch 2.5.1 con ROCm 6.2

**Virtual Environment (Raccomandato):**

```bash
# Crea venv condiviso su RAID
python3.12 -m venv /mnt/raid0/shared_envs/venv-rocm311
source /mnt/raid0/shared_envs/venv-rocm311/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

**Install PyTorch:**

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/rocm6.2
```

**Verifica:**

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Output: 2.5.1+rocm6.2

python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
# Output: CUDA Available: True

python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Output: GPU: AMD Radeon Instinct MI50
```

### PyTorch Configuration

**Memory Allocator (IMPORTANTE):**

```bash
# In start scripts o ~/.bashrc
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.95"
```

**Cosa fa:**
- `max_split_size_mb:128`: Limita dimensione frammenti VRAM (riduce frammentazione)
- `garbage_collection_threshold:0.95`: GC aggressivo (libera VRAM al 95% utilizzo)

**Attention Implementation:**

```bash
export OLLAMA_FAKE_ATTN_IMPL=sdpa
```

`sdpa` = Scaled Dot-Product Attention (ottimizzato per ROCm 5.x+)

---

## Python Environment

### Virtual Environment Condiviso

**Path:** `/mnt/raid0/shared_envs/venv-rocm311/`

**Perché condiviso?**
- Un solo venv per tutti i servizi → consistenza versioni
- Risparmio spazio (no duplicazione dipendenze)
- Installato su RAID → disponibile anche dopo reboot
- Facile upgrade centralizzato

### Creazione Venv

```bash
# Python 3.12 per MI50 services
python3.12 -m venv /mnt/raid0/shared_envs/venv-rocm311

# Activate
source /mnt/raid0/shared_envs/venv-rocm311/bin/activate

# Upgrade base tools
pip install --upgrade pip setuptools wheel
```

### Activation in Scripts

Tutti gli script di avvio devono includere:

```bash
#!/bin/bash
source /mnt/raid0/shared_envs/venv-rocm311/bin/activate
# ... resto script
```

O in systemd service:

```ini
[Service]
ExecStart=/mnt/raid0/shared_envs/venv-rocm311/bin/python /path/to/app.py
```

---

## Dipendenze Chiave

### Backend MI50 (requirements.txt)

```txt
# Core ML
torch==2.5.1
transformers==4.45.0
accelerate==0.34.2
sentencepiece==0.2.0
protobuf==5.28.2

# Web framework
fastapi==0.122.0
uvicorn[standard]==0.32.0
pydantic==2.10.1

# Utilities
python-multipart==0.0.12
aiofiles==24.1.0
python-dotenv==1.0.1
```

### Chat UI (requirements.txt)

```txt
fastapi==0.122.0
uvicorn[standard]==0.32.0
jinja2==3.1.4
requests==2.32.3
aiofiles==24.1.0
```

### Dashboard (requirements.txt)

```txt
flask==3.0.3
requests==2.32.3
streamlit==1.39.0  # Per monitoring avanzato
```

### RAG M40 (requirements.txt)

```txt
# Vector DB
chromadb==1.3.5
sentence-transformers==3.3.0

# ONNX Runtime (CPU)
onnxruntime==1.23.0

# Web framework
fastapi==0.122.0
uvicorn[standard]==0.32.0

# Utilities
numpy==1.26.4
pandas==2.2.3
```

### RAG Proxy (requirements.txt)

```txt
fastapi==0.122.0
uvicorn[standard]==0.32.0
requests==2.32.3
aiohttp==3.11.8
```

### Installazione Dipendenze

**Backend MI50:**

```bash
cd ~/mi50_stack/mi50_come_ollama
source /mnt/raid0/shared_envs/venv-rocm311/bin/activate
pip install -r requirements.txt
```

**Tutti i servizi:**

```bash
# Script per installare tutto
cd ~/mi50_stack
for service in mi50_come_ollama mi50_chat_ui mi50_dashboard; do
  cd ~/mi50_stack/$service
  pip install -r requirements.txt
done

cd ~/rag_di_sistema
pip install -r requirements.txt
```

---

## Modelli LLM

### Download da HuggingFace

**Configurazione cache:**

```bash
export HF_HOME=/mnt/raid0/hf_cache
export TRANSFORMERS_CACHE=/mnt/raid0/hf_cache/transformers
```

**Download manuale modello:**

```python
from huggingface_hub import snapshot_download

# Esempio: Qwen 2.5 Coder 7B
snapshot_download(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    local_dir="/mnt/raid0/qwen2.5-coder-7b-instruct",
    local_dir_use_symlinks=False
)
```

O via script bash:

```bash
cd /mnt/raid0
mkdir -p qwen2.5-coder-7b-instruct
cd qwen2.5-coder-7b-instruct

python3 << EOF
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    local_dir=".",
    local_dir_use_symlinks=False
)
EOF
```

### Modelli Disponibili

| Modello | Repo HuggingFace | Path Locale | VRAM |
|---------|------------------|-------------|------|
| **Qwen2.5-Coder-7B** | Qwen/Qwen2.5-Coder-7B-Instruct | /mnt/raid0/qwen2.5-coder-7b-instruct | 14GB |
| **Qwen2.5-Coder-14B** | Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4 | /mnt/raid0/qwen2.5-coder-14b-instruct | 28GB |
| **Gemma-2-9B-IT** | google/gemma-2-9b-it | /mnt/raid0/gemma-2-9b-it | 18GB |
| **DeepSeek-Coder-6.7B** | deepseek-ai/deepseek-coder-6.7b-instruct | /mnt/raid0/deepseek-coder-6.7b-instruct | 13GB |
| **Gemma3-4B-IT** | google/gemma-3-4b-it | /mnt/raid0/gemma3-4b-it | 8GB |

### Formato Modelli

Tutti i modelli sono in formato **FP16 (float16)** nativo o **GPTQ-Int4** per i modelli 14B+.

**Verifica formato:**

```bash
ls -lh /mnt/raid0/qwen2.5-coder-7b-instruct/*.safetensors
# Se file ~14GB → FP16
# Se file ~7GB → GPTQ-Int4
```

### Model Card Check

**Verifica compatibilità modello:**

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("/mnt/raid0/qwen2.5-coder-7b-instruct")
print(f"Model Type: {config.model_type}")
print(f"Hidden Size: {config.hidden_size}")
print(f"Num Layers: {config.num_hidden_layers}")
print(f"Vocab Size: {config.vocab_size}")
```

Output atteso (Qwen 7B):
```
Model Type: qwen2
Hidden Size: 3584
Num Layers: 28
Vocab Size: 151936
```

---

## Verifica Installazione

### Test Completo Stack

**Script di test: `~/test_stack.sh`**

```bash
#!/bin/bash

echo "=== System Info ==="
lsb_release -d
uname -r

echo -e "\n=== ROCm ==="
rocm-smi --version 2>/dev/null || echo "ROCm not found"

echo -e "\n=== PyTorch ==="
python3 << EOF
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except Exception as e:
    print(f"Error: {e}")
EOF

echo -e "\n=== Transformers ==="
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || echo "Not installed"

echo -e "\n=== FastAPI ==="
python3 -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')" 2>/dev/null || echo "Not installed"

echo -e "\n=== ONNX Runtime ==="
python3 -c "import onnxruntime; print(f'ONNX: {onnxruntime.__version__}')" 2>/dev/null || echo "Not installed"

echo -e "\n=== ChromaDB ==="
python3 -c "import chromadb; print(f'ChromaDB: {chromadb.__version__}')" 2>/dev/null || echo "Not installed"

echo -e "\n=== Disk Space ==="
df -h /mnt/raid0 | grep -v Filesystem

echo -e "\n=== RAM ==="
free -h | grep Mem

echo -e "\n=== GPU Status ==="
rocm-smi | grep -A1 "GPU  Temp"
```

Esegui:
```bash
chmod +x ~/test_stack.sh
./test_stack.sh
```

### Test Model Loading

**Test caricamento modello piccolo:**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/raid0/gemma3-4b-it"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16
)
model = model.to("cuda:0")

print("Testing inference...")
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda:0")
outputs = model.generate(**inputs, max_new_tokens=20)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Response: {response}")
print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

**Output atteso:**
```
Loading tokenizer...
Loading model...
Testing inference...
Response: Hello, how are you? I'm doing well, thank you for asking!
VRAM used: 8.24 GB
```

### Test API Endpoints

**Backend MI50:**

```bash
# Deve essere avviato prima
curl http://127.0.0.1:11534/api/version | python3 -m json.tool
```

**RAG M40:**

```bash
curl http://127.0.0.1:11600/health | python3 -m json.tool
```

---

## Troubleshooting Installation

### ROCm: GPU Not Detected

**Sintomo:**
```bash
rocm-smi
# Output: No GPU detected
```

**Fix:**

```bash
# Check driver loaded
lsmod | grep amdgpu
# Se vuoto:
sudo modprobe amdgpu

# Check device node
ls -la /dev/kfd /dev/dri/renderD*

# Reinstall ROCm se necessario
sudo amdgpu-install --usecase=rocm --opencl=rocr
```

### PyTorch: torch.cuda.is_available() = False

**Cause possibili:**
1. ROCm non installato
2. HIP_VISIBLE_DEVICES sbagliato
3. User non in gruppo video/render

**Fix:**

```bash
# Check groups
groups
# Deve includere: video render

# Add se mancante
sudo usermod -a -G video,render $USER
# Logout/login

# Check HIP
export HIP_VISIBLE_DEVICES=0
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Transformers: Model Loading Error

**Sintomo:**
```
OSError: Can't load model file...
```

**Cause:**
- Modello non completamente scaricato
- Cache corrotta
- Permessi errati

**Fix:**

```bash
# Check model files
ls -lh /mnt/raid0/qwen2.5-coder-7b-instruct/
# Deve contenere: config.json, *.safetensors, tokenizer*

# Clear cache
rm -rf /mnt/raid0/hf_cache/transformers/*

# Re-download
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Coder-7B-Instruct', local_dir='/mnt/raid0/qwen2.5-coder-7b-instruct', local_dir_use_symlinks=False)"
```

### ONNX Runtime: ImportError

**Sintomo:**
```
ImportError: libgomp.so.1: cannot open shared object file
```

**Fix:**

```bash
sudo apt install libgomp1
```

### ChromaDB: SQLite Error

**Sintomo:**
```
sqlite3.OperationalError: unable to open database file
```

**Fix:**

```bash
# Create data directory
mkdir -p ~/rag_di_sistema/chroma_db

# Fix permissions
chmod 755 ~/rag_di_sistema/chroma_db
```

---

## Next Steps

Con il software stack installato e verificato, procedi a:

**→ [03-backend-setup.md](./03-backend-setup.md)** - Configurazione Backend MI50 LLM

---

**Licenza:** CC BY-NC-SA 4.0
**Autore:** Raffaele Spezia
**Repository:** https://github.com/RaffaeleSpezia/local-llm-inference-lab
