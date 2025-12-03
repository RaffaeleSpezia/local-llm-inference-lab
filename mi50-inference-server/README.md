# MI50 Inference Server - Complete LLM Stack

**Complete production-ready LLM inference system running on AMD MI50 GPU (32GB VRAM)**

![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-blue)
![Python](https://img.shields.io/badge/python-3.12-blue)
![ROCm](https://img.shields.io/badge/ROCm-6.2.4-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+rocm6.2-red)

---

## Overview

Production-ready inference system for large language models optimized for **AMD MI50 GPU**. Includes backend server, web chat interface, monitoring dashboard, and complete documentation.

**Key Features:**
- ✅ **Ollama-compatible API** - Drop-in replacement for Ollama
- ✅ **Multi-model support** - Qwen, Gemma, DeepSeek (7B-14B)
- ✅ **Optimized VRAM usage** - 50% idle (16GB free for KV cache)
- ✅ **High throughput** - 50-100 tokens/sec on Qwen 7B
- ✅ **Streaming responses** - Real-time NDJSON streaming
- ✅ **Tool calling** - OpenAI-style function calling
- ✅ **Web interfaces** - Chat UI + Admin dashboard
- ✅ **Systemd services** - Auto-start on boot
- ✅ **Complete docs** - 8 comprehensive guides

---

## Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **GPU** | AMD MI50 32GB | AMD MI50 32GB | ROCm 6.2+ required |
| **RAM** | 64GB | 168GB+ | Used as VRAM optimization buffer |
| **Storage** | 200GB | 500GB SSD/NVMe | For models + cache |
| **CPU** | 8 cores | 16+ cores | For tokenization |
| **Network** | 1Gbps LAN | 10Gbps LAN | For remote API access |

**Tested on:**
- AMD Radeon Instinct MI50 32GB HBM2
- Ubuntu 22.04 LTS
- ROCm 6.2.4
- 168GB RAM (optimal for RAM-first loading strategy)

---

## Quick Start

### 1. Install Dependencies

```bash
# Install ROCm
sudo amdgpu-install --usecase=rocm

# Install PyTorch ROCm
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2

# Install dependencies
cd mi50-inference-server/backend
pip install -r requirements.txt
```

### 2. Download Model

```bash
cd /mnt/raid0
python3 << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    local_dir="/mnt/raid0/qwen2.5-coder-7b-instruct",
    local_dir_use_symlinks=False
)
EOF
```

### 3. Start Backend

```bash
cd mi50-inference-server/backend
source /mnt/raid0/shared_envs/venv-rocm311/bin/activate
export HIP_VISIBLE_DEVICES=0
python app.py /mnt/raid0/qwen2.5-coder-7b-instruct --port 11534
```

### 4. Test

```bash
# API test
curl http://localhost:11534/api/version

# Generation test
curl -X POST http://localhost:11534/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Write a Python hello world", "options": {"max_new_tokens": 50}}'
```

### 5. Start Chat UI

```bash
cd mi50-inference-server/chat-ui
export MI50_SERVER_URL="http://127.0.0.1:11534"
uvicorn app.main:app --host 0.0.0.0 --port 12000
```

Open browser: `http://192.168.1.155:12000`

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           User (Browser/API Client)                 │
└────────┬─────────────────────────────────┬──────────┘
         │                                 │
    ┌────▼─────┐                      ┌───▼───────┐
    │ Chat UI  │                      │ Direct    │
    │ (12000)  │                      │ API       │
    └────┬─────┘                      │ (11534)   │
         │                            └───┬───────┘
         │                                │
    ┌────▼────────────────────────────────▼─────────┐
    │     Backend MI50 (FastAPI)                    │
    │  - Model Manager                              │
    │  - Generation Engine                          │
    │  - Tool Calling                               │
    │  - Session Management                         │
    └────────────┬──────────────────────────────────┘
                 │
    ┌────────────▼──────────────────────────────────┐
    │   PyTorch + ROCm                              │
    │  - Transformers 4.45                          │
    │  - Model: Qwen2.5 Coder 7B                    │
    │  - FP16, Context: 32768 tokens                │
    └────────────┬──────────────────────────────────┘
                 │
    ┌────────────▼──────────────────────────────────┐
    │   AMD MI50 GPU                                │
    │  - 32GB VRAM (50% idle, 70% inference)        │
    │  - 50-100 tokens/sec                          │
    └───────────────────────────────────────────────┘
```

---

## Services

### Backend MI50 (Port 11534)

**LLM inference server (Ollama-compatible)**
- Models: Qwen 7B/14B, Gemma 2/3, DeepSeek 6.7B
- API: `/api/generate`, `/api/chat`, `/api/tags`
- Streaming: NDJSON real-time
- Tool calling: OpenAI-style
- VRAM: Optimized 50% idle

### Chat UI (Port 12000)

**Web interface for chat**
- Multi-session management
- Model selector
- Parameter tuning (temp, top-k, max tokens)
- Token counter + context trimming
- Export/import conversations

### Dashboard (Port 13000)

**Monitoring & admin UI**
- VRAM usage real-time
- Model status
- Service health checks
- Performance metrics

---

## Documentation

**Complete setup and troubleshooting guides:**

| Doc | Description | Link |
|-----|-------------|------|
| **01-hardware-setup.md** | GPU, RAM, Storage, Network configuration | [docs/01-hardware-setup.md](docs/01-hardware-setup.md) |
| **02-software-stack.md** | ROCm, PyTorch, Python deps installation | [docs/02-software-stack.md](docs/02-software-stack.md) |
| **03-backend-setup.md** | Backend configuration, API endpoints | [docs/03-backend-setup.md](docs/03-backend-setup.md) |
| **04-chat-ui-setup.md** | Web interface setup and customization | [docs/04-chat-ui-setup.md](docs/04-chat-ui-setup.md) |
| **05-dashboard-setup.md** | Monitoring dashboard setup | [docs/05-dashboard-setup.md](docs/05-dashboard-setup.md) |
| **06-systemd-services.md** | Auto-start services with systemd | [docs/06-systemd-services.md](docs/06-systemd-services.md) |
| **07-troubleshooting.md** | Common issues and solutions | [docs/07-troubleshooting.md](docs/07-troubleshooting.md) |
| **08-performance-tuning.md** | Advanced optimization techniques | [docs/08-performance-tuning.md](docs/08-performance-tuning.md) |

---

## Performance

**Benchmark (Qwen 2.5 Coder 7B FP16 on MI50):**

| Metric | Value | Notes |
|--------|-------|-------|
| **VRAM Idle** | 50% (16GB) | 16GB free for KV cache |
| **VRAM Inference** | 60-70% (20-24GB) | Depends on context length |
| **Throughput** | 50-100 tok/s | Temperature 0.0 (greedy) |
| **Latency (TTFT)** | 2-5s | Short prompts (<100 tokens) |
| **Context Length** | 32768 tokens | Qwen 2.5 max |
| **GPU Temp** | 40-50°C | Under load |
| **Power Draw** | 120-150W | During inference |

**Optimization highlights:**
- ✅ **RAM-first loading** - Reduces VRAM waste from 90% to 50%
- ✅ **Aggressive GC** - `garbage_collection_threshold:0.95`
- ✅ **GPU perf high** - Maximizes clock speeds
- ✅ **SDPA attention** - Optimized for ROCm 6.x

---

## Supported Models

| Model | VRAM | Tok/s | Quality | Use Case |
|-------|------|-------|---------|----------|
| **Qwen2.5 Coder 7B** | 14GB | 100 | ⭐⭐⭐⭐ | General purpose (recommended) |
| **Qwen2.5 Coder 14B GPTQ** | 14GB | 75 | ⭐⭐⭐⭐ | Quality + Speed |
| **Gemma 2 9B IT** | 18GB | 70 | ⭐⭐⭐⭐ | Balanced |
| **DeepSeek Coder 6.7B** | 13GB | 90 | ⭐⭐⭐⭐ | Code generation |
| **Gemma3 4B IT** | 8GB | 120 | ⭐⭐⭐ | Fast chat |

---

## API Examples

### Text Generation (Sync)

```bash
curl -X POST http://192.168.1.155:11534/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Write a Python function to reverse a string",
    "options": {
      "max_new_tokens": 512,
      "temperature": 0.7,
      "top_p": 0.9
    }
  }'
```

### Streaming Response

```bash
curl -N -X POST http://192.168.1.155:11534/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Explain recursion",
    "stream": true,
    "options": {"max_new_tokens": 200}
  }'
```

### Chat with Context

```bash
curl -X POST http://192.168.1.155:11534/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant"},
      {"role": "user", "content": "Write a quicksort in Python"}
    ],
    "stream": false
  }'
```

### Tool Calling

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
          }
        }
      }
    }]
  }'
```

---

## Troubleshooting

### VRAM at 90% Idle

**Fixed in October 2025.** If you encounter this, check that `model_manager.py` does **NOT** contain:
- `device_map="auto"`
- `low_cpu_mem_usage=True`

Should contain:
```python
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model = model.to("cuda:0")
torch.cuda.empty_cache()
```

See: [docs/07-troubleshooting.md](docs/07-troubleshooting.md)

### Generation Slow (<10 tok/s)

1. Set GPU performance: `sudo rocm-smi --setperflevel high`
2. Check VRAM: `rocm-smi` (should be <70%)
3. Use smaller model (7B instead of 14B)

See: [docs/08-performance-tuning.md](docs/08-performance-tuning.md)

---

## Directory Structure

```
mi50-inference-server/
├── README.md              # This file
├── LICENSE.md             # CC BY-NC-SA 4.0 license
│
├── docs/                  # Complete documentation (8 guides)
│   ├── 01-hardware-setup.md
│   ├── 02-software-stack.md
│   ├── 03-backend-setup.md
│   ├── 04-chat-ui-setup.md
│   ├── 05-dashboard-setup.md
│   ├── 06-systemd-services.md
│   ├── 07-troubleshooting.md
│   └── 08-performance-tuning.md
│
├── backend/               # Backend MI50 LLM service
│   ├── app.py             # FastAPI server
│   ├── model_manager.py   # Model loading + inference
│   ├── session_manager.py # Chat sessions
│   ├── tool_manager.py    # Tool calling
│   ├── rag_manager.py     # RAG integration (legacy)
│   ├── utils.py           # Utilities
│   ├── requirements.txt   # Python dependencies
│   └── README.md          # Backend-specific docs
│
├── chat-ui/               # Web chat interface
│   ├── app/
│   │   ├── main.py        # FastAPI app
│   │   ├── storage.py     # Chat persistence
│   │   ├── prompt_formatter.py  # Model-specific formats
│   │   └── token_counter.py     # Token counting
│   ├── static/
│   │   └── index.html     # Single-page UI
│   ├── start_chat_ui.sh   # Start script
│   └── requirements.txt
│
├── dashboard/             # Monitoring dashboard
│   ├── app/
│   │   └── dashboard.py   # Flask app
│   ├── static/
│   │   └── index.html
│   ├── start_dashboard.sh
│   └── requirements.txt
│
├── scripts/               # Management scripts
│   ├── start_all_services.sh    # Start all (backend+ui+dashboard)
│   ├── stop_all_services.sh     # Stop all
│   ├── status_services.sh       # Check status
│   └── start_mi50_ai_server.sh  # Interactive model selector
│
├── systemd/               # Systemd service files
│   ├── mi50-backend.service
│   ├── mi50-chat-ui.service
│   ├── mi50-dashboard.service
│   └── mi50-stack.target
│
└── nginx/                 # Nginx config (optional reverse proxy)
    └── mi50-stack.conf
```

---

## Development

**Testing changes:**

```bash
# Backend test
cd backend
pytest tests/

# Smoke test
python smoke_test.py --host http://localhost:11534 --prompt "Hello"

# Load test
ab -n 100 -c 10 -p payload.json -T application/json http://localhost:11534/api/generate
```

**Adding new model:**

1. Download model to `/mnt/raid0/`
2. Update `start_mi50_ai_server.sh` (add to model list)
3. Update `chat-ui/app/prompt_formatter.py` (add format)
4. Test: `python app.py /mnt/raid0/new-model --port 11534`

---

## License

**CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International)

See [LICENSE.md](LICENSE.md) for full details.

**Commercial licensing available.** Contact: lele.sra@gmail.com

---

## Author

**Raffaele Spezia**
- Email: lele.sra@gmail.com | info@axefactory.com
- GitHub: [@RaffaeleSpezia](https://github.com/RaffaeleSpezia)
- Repository: https://github.com/RaffaeleSpezia/local-llm-inference-lab

---

## Acknowledgments

- AMD ROCm team for GPU compute platform
- HuggingFace Transformers for model loading
- Qwen, Gemma, DeepSeek teams for excellent LLMs
- FastAPI and Uvicorn for high-performance web framework

---

## Related Projects

- **[ai-consciousness-research](https://github.com/RaffaeleSpezia/ai-consciousness-research)** - AI consciousness protocols
- **[functional-autonomy-manual](https://github.com/RaffaeleSpezia/functional-autonomy-manual)** - Functional autonomy framework

---

**Status:** Production Ready ✅
**Version:** 2.0
**Last Updated:** December 3, 2025
