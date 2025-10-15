# ROCm-Ollama-Bypass

FastAPI service that mirrors the Ollama HTTP API but runs models directly with PyTorch/Transformers on AMD GPUs (ROCm). Useful when the official Ollama binary does not support your cards (e.g. MI50/M40, Vega) or when you need full precision weights.

## Features
- `/api/generate` and `/api/chat` with NDJSON streaming, stop sequences, session memory.
- `/api/version`, `/api/tags`, `/api/show`, `/api/delete`, `/api/pull` stubs for drop-in compatibility.
- Optional Retrieval-Augmented Generation (`/rag/upsert`, `/rag/query`) powered by FAISS + MiniLM embeddings.
- Simple `start.sh` helper that configures the virtualenv, cache paths and launches Uvicorn.
- Structured JSON logging with optional rotating files, smoke tests and FastAPI unit tests.

## Requirements
- ROCm-enabled PyTorch build (`torch>=2.1` compiled with HIP) and Python 3.10+.
- A cached HuggingFace model in full precision (`bf16`/`fp16`). Example: `Qwen/Qwen2.5-7B-Instruct`.
- Recommended: dedicated cache directories on fast storage (NVMe/RAID) to avoid repeated downloads.

## Quick Start
```bash
# 1. Clone or copy this directory
cd ROCm-Ollama-Bypass

# 2. Create / activate a ROCm-ready venv (example)
python3 -m venv ~/venvs/rocm-ollama
source ~/venvs/rocm-ollama/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Point caches to fast storage (optional but recommended)
export HF_HOME=/mnt/raid/hf_cache
export TRANSFORMERS_CACHE=/mnt/raid/hf_cache/transformers
export TORCH_HOME=/mnt/raid/torch_cache

# 4. Launch (defaults to port 11534 to avoid conflict with official Ollama)
./start.sh "/path/to/your/hf_model"
```

`start.sh` checks that the virtualenv exists, validates port availability (both `lsof` and `ss`) and exports basic env vars (`OLLAMA_FAKE_LOGDIR`, `OLLAMA_FAKE_LOGLEVEL`, caches). Override defaults with environment variables, e.g. `PORT=11434 ./start.sh`.

## Manual Launch
```bash
source ~/venvs/rocm-ollama/bin/activate
export HF_HOME=/mnt/raid/hf_cache
export TRANSFORMERS_CACHE=/mnt/raid/hf_cache/transformers
export TORCH_HOME=/mnt/raid/torch_cache
python app.py "/path/to/model" --host 0.0.0.0 --port 11534 --log-level info
```

## API Compatibility
- Send the same JSON payloads you would send to Ollama.
- Generation parameters go inside `"options"`: `temperature`, `top_p`, `top_k`, `seed`, `num_predict`, `repeat_penalty`, `stop`, etc.
- `/api/generate` supports `"stream": true` for NDJSON streaming.
- `/api/chat` accepts `session_id`; memory is kept in RAM (swap out `SessionManager` if you need persistence).

Example:
```bash
curl -N http://127.0.0.1:11534/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
        "prompt": "Dammi tre punti sulla GPU MI50",
        "stream": true,
        "options": {"temperature": 0.7, "top_p": 0.9, "seed": 42}
      }'
```

## Systemd Service
`systemd/mi50_ollama.service` is a template. Edit `WorkingDirectory`, `ExecStart` and the model path, then:
```bash
sudo cp systemd/mi50_ollama.service /etc/systemd/system/mi50_ollama.service
sudo systemctl daemon-reload
sudo systemctl enable --now mi50_ollama.service
```
Logs go to `logs/mi50_ollama.log` by default (JSON lines). Update `OLLAMA_FAKE_LOGDIR` if you prefer a different path.

## Tests
```bash
./run_tests.sh                # runs pytest inside the current venv
python smoke_test.py --host http://127.0.0.1:11534 --model "/path/to/model"
```
`tests/test_app.py` uses monkeypatched dummy models so it runs quickly without GPU.

## Troubleshooting
- `amdgpu.ids: No such file or directory` is a benign ROCm warning; create `/etc/amdgpu/ids/amdgpu.ids` if you want to silence it.
- If you see `Errno 98 ... address already in use`, another server is using the chosen port (maybe the official `ollama serve`). Either stop it or set `PORT=11535` before launching.
- First request can take several seconds while the model loads into VRAM. Subsequent requests reuse the cached handle (`Model ... already loaded`).

## Roadmap
- Multi-model registry exposed via `/api/tags`.
- Persisted chat history backend (Redis / SQLite).
- Optional quantized loaders on CUDA (bitsandbytes) for hybrid MI50/M40 clusters.

Contributions welcome! Open issues or pull requests once this folder is pushed to GitHub.
