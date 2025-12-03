# Backend Setup - MI50 LLM Service

**Versione:** 1.0
**Data:** 3 Dicembre 2025
**Autore:** Raffaele Spezia

---

## Indice

1. [Overview Backend](#overview-backend)
2. [Struttura Codice](#struttura-codice)
3. [Configurazione](#configurazione)
4. [Model Manager](#model-manager)
5. [API Endpoints](#api-endpoints)
6. [Tool Calling](#tool-calling)
7. [RAG Integration](#rag-integration)
8. [Avvio Servizio](#avvio-servizio)
9. [Monitoring](#monitoring)

---

## Overview Backend

### Che Cos'è

Il **Backend MI50** è un server REST API **compatibile Ollama** che:
- Carica modelli LLM da HuggingFace su GPU AMD MI50
- Espone endpoint `/api/generate`, `/api/chat`, `/api/tags` compatibili Ollama
- Supporta streaming NDJSON per risposte in tempo reale
- Implementa tool calling stile OpenAI
- Integra RAG opzionale per retrieval-augmented generation

### Tecnologie

| Componente | Tecnologia | Ruolo |
|------------|-----------|-------|
| **Web Framework** | FastAPI 0.122 | REST API async |
| **Model Loading** | Transformers 4.45 | HuggingFace models |
| **Inference Engine** | PyTorch 2.5.1+rocm6.2 | GPU compute |
| **GPU Runtime** | ROCm 6.2.4 HIP | AMD MI50 |
| **Tokenization** | Tokenizers (Rust) | Fast tokenizer |
| **Streaming** | TextIteratorStreamer | Real-time generation |

### Architettura

```
┌─────────────────────────────────────────────────────┐
│                   app.py (FastAPI)                  │
│  ┌───────────────────────────────────────────────┐  │
│  │ Endpoint Layer                                │  │
│  │  /api/generate, /api/chat, /api/tags, ...    │  │
│  └────────┬──────────────────────────────────────┘  │
│           │                                          │
│  ┌────────▼──────────────────────────────────────┐  │
│  │ Request Parsing & Validation (Pydantic)      │  │
│  └────────┬──────────────────────────────────────┘  │
│           │                                          │
│  ┌────────▼──────────────────────────────────────┐  │
│  │ model_manager.py                             │  │
│  │  - load_model()                               │  │
│  │  - generate_sync()                            │  │
│  │  - generate_stream()                          │  │
│  └────────┬──────────────────────────────────────┘  │
│           │                                          │
│  ┌────────▼──────────────────────────────────────┐  │
│  │ PyTorch + Transformers                        │  │
│  │  AutoModelForCausalLM.generate()              │  │
│  └────────┬──────────────────────────────────────┘  │
│           │                                          │
│  ┌────────▼──────────────────────────────────────┐  │
│  │ ROCm HIP                                      │  │
│  │  GPU kernels, memory allocation               │  │
│  └────────┬──────────────────────────────────────┘  │
└───────────┼──────────────────────────────────────────┘
            │
     ┌──────▼─────┐
     │  MI50 GPU  │
     │  32GB VRAM │
     └────────────┘
```

---

## Struttura Codice

### File Principali

```
mi50_come_ollama/
├── app.py                      # FastAPI server principale (900+ righe)
├── model_manager.py            # Caricamento modelli + inferenza (400 righe)
├── session_manager.py          # Gestione sessioni chat stateful
├── tool_manager.py             # Tool calling OpenAI-style
├── rag_manager.py              # RAG store (deprecato, ora usa M40)
├── token_broadcaster.py        # Broadcasting token streaming
├── utils.py                    # Logger, helpers vari
│
├── start.sh                    # Script avvio interattivo
├── requirements.txt            # Dipendenze Python
│
├── systemd/
│   └── mi50_ollama.service    # Systemd service file
│
└── scripts/
    └── flush_logs.sh           # Flush log da ramdisk a RAID
```

### app.py - Server FastAPI

**Responsabilità:**
- Definizione endpoint REST
- Validazione request con Pydantic
- Dispatch a model_manager
- Streaming response NDJSON
- Tool calling orchestration
- Logging dettagliato

**Punti chiave:**

```python
# Line 174-206: Application factory
def create_app() -> FastAPI:
    app = FastAPI(title="Ollama-Compatible PyTorch Service", version="0.1.0")

    # State initialization
    app.state.model_manager = ModelManager(device="cuda:0", dtype=torch.float16)
    app.state.session_manager = SessionManager()
    app.state.token_broadcaster = TokenBroadcaster()

    return app

# Line 81-90: VRAM monitoring
def get_vram_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return {"allocated_gb": round(allocated, 2), "reserved_gb": round(reserved, 2)}
    return {"allocated_gb": 0, "reserved_gb": 0}
```

### model_manager.py - Model Loading

**Responsabilità:**
- Load/unload modelli HuggingFace
- Ottimizzazione VRAM (strategia RAM-first)
- Generazione testo sync/stream
- Gestione tokenizer

**Punti critici:**

```python
# Line 120-206: Load model with RAM-first strategy
def load_model(self, model_name: str, quantize: Optional[str] = None) -> ModelHandle:
    # Unload previous models (MI50 = single model at a time)
    if self._models:
        for old_model in list(self._models.keys()):
            handle = self._models.pop(old_model)
            del handle.model
            del handle.tokenizer
        torch.cuda.empty_cache()

    # Load to RAM first (168GB available)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=self.dtype  # NO device_map, NO low_cpu_mem_usage
    )

    # Transfer RAM → VRAM
    model = model.to("cuda:0")

    # Critical: Free temporary buffers
    torch.cuda.empty_cache()

    return handle
```

**Perché questa strategia?**

❌ **Approccio classico (causava problema VRAM 90%):**
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",           # ❌ Alloca buffer VRAM temporanei
    low_cpu_mem_usage=True       # ❌ Carica chunk-by-chunk → doppia allocazione
)
# Risultato: 14GB modello + 14GB buffer = 28GB sprecati
```

✅ **Approccio ottimizzato (VRAM 50%):**
```python
# 1. Load intero modello in RAM (15GB su 168GB = 9% utilizzo)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)

# 2. Transfer diretto RAM → VRAM (no buffer intermedi)
model = model.to("cuda:0")

# 3. Cleanup buffer temporanei
torch.cuda.empty_cache()

# Risultato: 14GB VRAM, 16GB liberi per KV cache
```

---

## Configurazione

### Variabili Ambiente

**File:** `start.sh` o `~/.bashrc`

#### GPU e Memoria

```bash
# GPU selection (MI50 = device 0)
export HIP_VISIBLE_DEVICES=0

# PyTorch memory allocator
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.95"
```

#### Cache HuggingFace/Torch

```bash
# Cache condivise su RAID
export HF_HOME=/mnt/raid0/hf_cache
export TRANSFORMERS_CACHE=/mnt/raid0/hf_cache/transformers
export TORCH_HOME=/mnt/raid0/torch_cache
```

#### Server Settings

```bash
# Modello default se non specificato
export OLLAMA_FAKE_DEFAULT_MODEL="/mnt/raid0/qwen2.5-coder-7b-instruct"

# Logging
export OLLAMA_FAKE_LOGLEVEL="info"  # debug | info | warning | error
export OLLAMA_FAKE_LOGDIR="/dev/shm/mi50_ollama_logs"  # Ramdisk per performance
export OLLAMA_FAKE_LOG_TOKENS="true"  # Log streaming token
export OLLAMA_FAKE_LOG_TOKEN_CHARS="160"  # Limite caratteri log token
```

#### Generation Defaults

```bash
# Inference parameters (fast greedy by default)
export OLLAMA_FAKE_DEFAULT_MAX_NEW_TOKENS="128"
export OLLAMA_FAKE_DEFAULT_TEMPERATURE="0.0"  # Greedy decoding
export OLLAMA_FAKE_DEFAULT_TOP_K="0"
export OLLAMA_FAKE_DEFAULT_TOP_P="0.95"
export OLLAMA_FAKE_DEFAULT_REPETITION_PENALTY="1.0"

# Prompt limits
export OLLAMA_FAKE_MAX_PROMPT_TOKENS="4096"

# Streaming
export OLLAMA_FAKE_STREAM_CHARS="160"  # Chunk size (0 = token-by-token)
```

#### Attention Implementation

```bash
# Ottimizzato per ROCm 5.x+
export OLLAMA_FAKE_ATTN_IMPL="sdpa"  # Scaled Dot-Product Attention
```

### File start.sh

**Script di avvio interattivo:**

```bash
#!/bin/bash

# Activate venv
source /mnt/raid0/shared_envs/venv-rocm311/bin/activate

# Export environment
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.95"
export HF_HOME=/mnt/raid0/hf_cache
export TRANSFORMERS_CACHE=/mnt/raid0/hf_cache/transformers
export TORCH_HOME=/mnt/raid0/torch_cache
export OLLAMA_FAKE_LOGDIR="/dev/shm/mi50_ollama_logs"
export OLLAMA_FAKE_DEFAULT_MODEL="/mnt/raid0/qwen2.5-coder-7b-instruct"
export OLLAMA_FAKE_DEFAULT_MAX_NEW_TOKENS="128"
export OLLAMA_FAKE_DEFAULT_TEMPERATURE="0.0"
export OLLAMA_FAKE_ATTN_IMPL="sdpa"

# Create log dir
mkdir -p "$OLLAMA_FAKE_LOGDIR"

# Check if port 11534 is busy
if lsof -iTCP:11534 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "⚠️  Port 11534 already in use"
    echo "Kill existing process? (y/n)"
    read -r answer
    if [ "$answer" = "y" ]; then
        pkill -f "python.*app.py"
        sleep 2
    else
        echo "Aborted"
        exit 1
    fi
fi

# Start server
echo "Starting MI50 Backend on port 11534..."
python app.py "$OLLAMA_FAKE_DEFAULT_MODEL" --host 0.0.0.0 --port 11534 --log-level info
```

---

## Model Manager

### Load/Unload Lifecycle

**1. First Load:**

```bash
# Request: /api/generate with model="/mnt/raid0/qwen2.5-coder-7b-instruct"
```

```python
# In model_manager.py
handle = load_model("/mnt/raid0/qwen2.5-coder-7b-instruct")
# Steps:
# 1. Check if already loaded → No
# 2. Load tokenizer (0.5GB RAM)
# 3. Load model to RAM (14GB → 168GB, 8% usage)
# 4. Transfer to VRAM cuda:0 (14GB)
# 5. Empty cache (free temp buffers)
# 6. Store handle in _models dict
```

**2. Subsequent Requests:**

```python
# Same model → reuse handle
handle = get_handle("/mnt/raid0/qwen2.5-coder-7b-instruct")
# No reload, instant response
```

**3. Model Switch:**

```bash
# Request: /api/generate with model="/mnt/raid0/gemma-2-9b-it"
```

```python
# In load_model():
if self._models:  # Previous model exists
    # Unload old model
    old_handle = self._models.pop(old_model_name)
    del old_handle.model
    del old_handle.tokenizer
    torch.cuda.empty_cache()  # Free VRAM
    # → VRAM drops from 50% to ~2%

# Load new model
# → VRAM goes to ~55% (18GB Gemma 2 9B)
```

### Generation Methods

**generate_sync() - Blocking**

```python
# For: stream=false requests
text = model_manager.generate_sync(
    model_name="/mnt/raid0/qwen2.5-coder-7b-instruct",
    prompt="Write a Python hello world",
    options=GenerationOptions(
        max_new_tokens=128,
        temperature=0.0,
        top_k=0,
        top_p=0.95
    )
)
# Returns: Complete text string
```

**generate_stream() - Iterator**

```python
# For: stream=true requests
for chunk in model_manager.generate_stream(
    model_name="/mnt/raid0/qwen2.5-coder-7b-instruct",
    prompt="Write a Python hello world",
    options=...
):
    # Yields: Chunks of text (~160 chars each)
    yield chunk
```

**Chunking Strategy:**

```python
# In model_manager.py Line 282-305
buffer = ""
for token_text in streamer:  # Token-by-token from PyTorch
    buffer += token_text

    # Flush buffer when:
    should_flush = (
        "\n" in buffer or                        # Newline found
        len(buffer) >= stream_chunk_chars or     # Buffer full (160 chars)
        end_of_stream                             # Generation done
    )

    if should_flush:
        yield buffer
        buffer = ""
```

**Perché non token-by-token?**
- JSON serialization overhead: 5-10x latency per token
- Network overhead: TCP packets per token
- Chunk di 160 chars = ~40 token = buon compromesso velocità/latenza

---

## API Endpoints

### Base URL

```
http://192.168.1.155:11534
```

### GET /api/version

**Descrizione:** Info su servizio + GPU

**Response:**

```json
{
  "version": "0.1.0",
  "device": "cuda:0",
  "gpu_name": "AMD Radeon Instinct MI50",
  "dtype": "torch.float16",
  "quantization_supported": false
}
```

**Esempio:**

```bash
curl http://192.168.1.155:11534/api/version | jq
```

### GET /api/tags

**Descrizione:** Lista modelli caricati

**Response:**

```json
{
  "models": [
    {
      "name": "/mnt/raid0/qwen2.5-coder-7b-instruct",
      "size": 7616000000,
      "digest": "2025-12-03T10:30:45Z",
      "details": {
        "context_length": 32768,
        "num_parameters": 7616000000,
        "dtype": "torch.float16",
        "device": "cuda:0"
      }
    }
  ]
}
```

**Esempio:**

```bash
curl http://192.168.1.155:11534/api/tags | jq
```

### POST /api/generate

**Descrizione:** Text generation (sync o streaming)

**Request Body:**

```json
{
  "prompt": "Write a Python function to reverse a string",
  "model": "/mnt/raid0/qwen2.5-coder-7b-instruct",
  "stream": false,
  "options": {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "stop": ["\n\n", "```"]
  }
}
```

**Parameters:**

| Campo | Tipo | Default | Descrizione |
|-------|------|---------|-------------|
| `prompt` | string | required | Input text |
| `model` | string | env default | Model path |
| `stream` | boolean | false | NDJSON streaming |
| `options.max_new_tokens` | int | 128 | Max tokens generati |
| `options.temperature` | float | 0.0 | Sampling temp (0=greedy) |
| `options.top_k` | int | 0 | Top-K sampling (0=off) |
| `options.top_p` | float | 0.95 | Nucleus sampling |
| `options.repetition_penalty` | float | 1.0 | Penalty ripetizioni |
| `options.seed` | int | null | Random seed |
| `options.stop` | array | null | Stop sequences |

**Response (stream=false):**

```json
{
  "model": "/mnt/raid0/qwen2.5-coder-7b-instruct",
  "created_at": "2025-12-03T10:35:22Z",
  "response": "def reverse_string(s):\n    return s[::-1]\n\n# Example usage:\nprint(reverse_string('hello'))  # Output: olleh",
  "done": true,
  "total_duration": 2500000000,
  "load_duration": 0,
  "prompt_eval_count": 12,
  "eval_count": 45
}
```

**Response (stream=true) - NDJSON:**

```json
{"model": "...", "created_at": "...", "response": "def ", "done": false}
{"model": "...", "created_at": "...", "response": "reverse_string(s):\n", "done": false}
{"model": "...", "created_at": "...", "response": "    return s[::-1]\n", "done": false}
{"model": "...", "created_at": "...", "response": "", "done": true, "total_duration": 2500000000}
```

**Esempio sync:**

```bash
curl -X POST http://192.168.1.155:11534/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Explain recursion in 50 words",
    "options": {"max_new_tokens": 100, "temperature": 0.7}
  }' | jq -r '.response'
```

**Esempio streaming:**

```bash
curl -N -X POST http://192.168.1.155:11534/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Write a story about AI",
    "stream": true,
    "options": {"max_new_tokens": 300}
  }'
```

### POST /api/chat

**Descrizione:** Chat con context history (stateful con session_id)

**Request Body:**

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful coding assistant"},
    {"role": "user", "content": "Write a Python quicksort"}
  ],
  "model": "/mnt/raid0/qwen2.5-coder-7b-instruct",
  "stream": false,
  "session_id": "my-session-123",
  "options": {
    "max_new_tokens": 512,
    "temperature": 0.7
  }
}
```

**Parameters:**

| Campo | Tipo | Default | Descrizione |
|-------|------|---------|-------------|
| `messages` | array | required | Chat messages |
| `session_id` | string | null | Session ID (stateful) |
| `tools` | array | null | Tool definitions (OpenAI-style) |

**Response:**

```json
{
  "model": "/mnt/raid0/qwen2.5-coder-7b-instruct",
  "created_at": "2025-12-03T10:40:15Z",
  "message": {
    "role": "assistant",
    "content": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)"
  },
  "done": true
}
```

**Session Management:**

```bash
# First message in session
curl -X POST http://192.168.1.155:11534/api/chat \
  -d '{"session_id": "coding-session", "messages": [{"role": "user", "content": "Hello"}]}'

# Follow-up (history maintained)
curl -X POST http://192.168.1.155:11534/api/chat \
  -d '{"session_id": "coding-session", "messages": [{"role": "user", "content": "Explain the previous code"}]}'
# → Backend mantiene context "Hello" + risposta precedente
```

### GET /debug/memory

**Descrizione:** VRAM usage real-time

**Response:**

```json
{
  "vram_allocated_gb": 14.23,
  "vram_reserved_gb": 15.50,
  "vram_total_gb": 32.0,
  "vram_free_gb": 16.50,
  "waste_gb": 1.27
}
```

**Calcolo waste:**
```
waste = reserved - allocated
```
Se `waste > 2GB` → possibile memory leak

**Esempio:**

```bash
watch -n 1 'curl -s http://192.168.1.155:11534/debug/memory | jq'
```

---

## Tool Calling

### Abilitazione Tools

**Request con tools (OpenAI-style):**

```json
{
  "messages": [{"role": "user", "content": "List files in current directory"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "execute_command",
        "description": "Execute a shell command",
        "parameters": {
          "type": "object",
          "properties": {
            "command": {"type": "string", "description": "The command to run"}
          },
          "required": ["command"]
        }
      }
    }
  ],
  "stream": true
}
```

**Response con tool_calls:**

```json
{
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "id": "call_123",
        "type": "function",
        "function": {
          "name": "execute_command",
          "arguments": "{\"command\": \"ls -la\"}"
        }
      }
    ]
  },
  "done": true,
  "done_reason": "tool_calls"
}
```

### Tool Flow

```
1. Client → /api/chat con tools + user message
2. Backend → Format prompt con tool definitions
3. LLM → Generate tool_call JSON
4. Backend → Parse JSON, return tool_calls array
5. Client → Execute tool, get result
6. Client → /api/chat con tool result message
7. Backend → Generate final response
```

### Esempio Completo

**Step 1: User message + tools**

```bash
curl -X POST http://192.168.1.155:11534/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "What is 25 * 4?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "calculator",
        "description": "Perform math operations",
        "parameters": {
          "type": "object",
          "properties": {
            "expression": {"type": "string"}
          },
          "required": ["expression"]
        }
      }
    }]
  }'
```

**Response:**

```json
{
  "message": {
    "role": "assistant",
    "tool_calls": [{
      "id": "call_abc",
      "function": {
        "name": "calculator",
        "arguments": "{\"expression\": \"25 * 4\"}"
      }
    }]
  },
  "done_reason": "tool_calls"
}
```

**Step 2: Tool execution (client-side)**

```python
import json
result = eval("25 * 4")  # = 100 (in real app: safe execution)
```

**Step 3: Send tool result**

```bash
curl -X POST http://192.168.1.155:11534/api/chat \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 25 * 4?"},
      {"role": "assistant", "tool_calls": [...]},
      {"role": "tool", "name": "calculator", "content": "100"}
    ]
  }'
```

**Response:**

```json
{
  "message": {
    "role": "assistant",
    "content": "The result of 25 multiplied by 4 is 100."
  },
  "done": true
}
```

---

## RAG Integration

**Nota:** RAG è ora gestito da servizio separato (RAG M40 porta 11600). Backend MI50 mantiene supporto legacy.

### Endpoint RAG (Legacy)

**POST /rag/upsert**

```bash
curl -X POST http://192.168.1.155:11534/rag/upsert \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_id": "python-docs",
    "documents": [
      {"id": "doc1", "text": "Python is a high-level programming language..."},
      {"id": "doc2", "text": "List comprehensions provide a concise way..."}
    ]
  }'
```

**POST /rag/query**

```bash
curl -X POST http://192.168.1.155:11534/rag/query \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_id": "python-docs",
    "query": "How do list comprehensions work?",
    "top_k": 3
  }'
```

### RAG in Generate/Chat

```json
{
  "prompt": "Explain list comprehensions",
  "rag": {
    "dataset_id": "python-docs",
    "top_k": 3
  }
}
```

Backend:
1. Query RAG store con prompt
2. Retrieve top-k documenti simili
3. Prefix prompt con context:
```
Context:
[Retrieved document 1]
[Retrieved document 2]

User query: Explain list comprehensions
```
4. Generate con prompt augmented

---

## Avvio Servizio

### Manuale (Foreground)

```bash
ssh lele2@192.168.1.155
cd ~/mi50_stack/mi50_come_ollama

# Activate venv
source /mnt/raid0/shared_envs/venv-rocm311/bin/activate

# Export env
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.95"
export HF_HOME=/mnt/raid0/hf_cache

# Start
python app.py "/mnt/raid0/qwen2.5-coder-7b-instruct" --host 0.0.0.0 --port 11534 --log-level info
```

### Via Script (Background)

```bash
cd ~/mi50_stack/mi50_come_ollama
./start.sh
# → Chiede conferma se porta occupata
# → Avvia in foreground
```

O background con nohup:

```bash
cd ~/mi50_stack
./start_all_services.sh
# → Avvia backend + chat UI + dashboard in background
```

### Via Systemd

**Service file:** `/etc/systemd/system/mi50_backend.service`

```ini
[Unit]
Description=MI50 LLM Backend Service
After=network.target

[Service]
Type=simple
User=lele2
WorkingDirectory=/mnt/raid0/services/mi50_ollama_like
Environment="HIP_VISIBLE_DEVICES=0"
Environment="PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.95"
Environment="HF_HOME=/mnt/raid0/hf_cache"
Environment="TRANSFORMERS_CACHE=/mnt/raid0/hf_cache/transformers"
Environment="TORCH_HOME=/mnt/raid0/torch_cache"
Environment="OLLAMA_FAKE_LOGDIR=/dev/shm/mi50_ollama_logs"
Environment="OLLAMA_FAKE_DEFAULT_MODEL=/mnt/raid0/qwen2.5-coder-7b-instruct"
ExecStart=/mnt/raid0/shared_envs/venv-rocm311/bin/python app.py /mnt/raid0/qwen2.5-coder-7b-instruct --host 0.0.0.0 --port 11534
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

**Abilitazione:**

```bash
sudo cp systemd/mi50_backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable mi50_backend.service
sudo systemctl start mi50_backend.service

# Check status
sudo systemctl status mi50_backend.service

# Logs
journalctl -u mi50_backend.service -f
```

---

## Monitoring

### Logs

**Location:** `/dev/shm/mi50_ollama_logs/mi50_ollama.log` (ramdisk)

**Tail logs:**

```bash
tail -f /dev/shm/mi50_ollama_logs/mi50_ollama.log
```

**Log format (JSON):**

```json
{"timestamp": "2025-12-03T10:45:12Z", "level": "INFO", "message": "[/api/generate] Prompt length: 45 chars (~11 tokens)"}
{"timestamp": "2025-12-03T10:45:12Z", "level": "INFO", "message": "[/api/generate] VRAM before: 14.23GB allocated"}
{"timestamp": "2025-12-03T10:45:14Z", "level": "INFO", "message": "[stream] Generated 128 tokens in 2.1s (61 tok/s)"}
```

### VRAM Monitoring

```bash
# Real-time VRAM
watch -n 1 'curl -s http://192.168.1.155:11534/debug/memory | jq'

# Or via rocm-smi
watch -n 1 rocm-smi
```

### Health Check

```bash
# Quick check
curl http://192.168.1.155:11534/api/version

# Full test
curl -X POST http://192.168.1.155:11534/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Say hello", "options": {"max_new_tokens": 10}}'
```

---

## Next Steps

Con il backend configurato, procedi a:

**→ [04-chat-ui-setup.md](./04-chat-ui-setup.md)** - Setup interfaccia web chat

---

**Licenza:** CC BY-NC-SA 4.0
**Autore:** Raffaele Spezia
**Repository:** https://github.com/RaffaeleSpezia/local-llm-inference-lab
