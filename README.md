# Local LLM Inference Lab

**Experimental repository for local LLM inference systems and optimizations**

![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-blue)
![Status](https://img.shields.io/badge/status-active-green)

---

## Overview

This repository contains experimental projects and documentation for running large language models locally on various hardware configurations, with focus on:
- AMD GPU optimization (ROCm)
- NVIDIA GPU configurations
- VRAM optimization techniques
- Inference performance tuning
- Production deployment strategies

---

## Projects

### 🚀 MI50 Inference Server (Production Ready)

**Complete LLM inference stack on AMD MI50 GPU**

Full production system with backend, web UI, dashboard, and comprehensive documentation.

**Features:**
- Ollama-compatible REST API
- Multi-model support (Qwen, Gemma, DeepSeek)
- Optimized VRAM usage (50% idle vs 90% baseline)
- 50-100 tokens/sec throughput
- Web chat interface
- Monitoring dashboard
- Systemd services
- 8 comprehensive setup guides

**Hardware:** AMD MI50 32GB + 168GB RAM

**👉 [Go to MI50 Inference Server Documentation](mi50-inference-server/README.md)**

---

## Documentation Standards

All projects in this repository follow these documentation standards:

**Comprehensive Setup Guides:**
1. Hardware setup and requirements
2. Software stack installation
3. Service configuration
4. UI/Dashboard setup
5. Systemd service management
6. Troubleshooting common issues
7. Performance tuning
8. API reference

**Code Standards:**
- Python 3.11+ (3.12 preferred)
- Type hints where applicable
- FastAPI for REST APIs
- Async/await for I/O operations
- Comprehensive error handling
- JSON structured logging

---

## Repository Structure

```
local-llm-inference-lab/
├── README.md                    # This file
├── mi50-inference-server/       # MI50 GPU production stack
│   ├── README.md                # Project overview
│   ├── LICENSE.md               # CC BY-NC-SA 4.0
│   ├── docs/                    # 8 comprehensive guides
│   ├── backend/                 # FastAPI LLM service
│   ├── chat-ui/                 # Web chat interface
│   ├── dashboard/               # Monitoring UI
│   ├── scripts/                 # Management scripts
│   ├── systemd/                 # Service files
│   └── nginx/                   # Reverse proxy config
│
└── [future projects]/           # Additional experiments
```

---

## Getting Started

### MI50 Inference Server Quick Start

```bash
# Clone repository
git clone https://github.com/RaffaeleSpezia/local-llm-inference-lab.git
cd local-llm-inference-lab/mi50-inference-server

# Read setup guide
cat docs/01-hardware-setup.md

# Install dependencies
cd backend
pip install -r requirements.txt

# Download model
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Coder-7B-Instruct', local_dir='/mnt/raid0/qwen2.5-coder-7b-instruct')"

# Start backend
python app.py /mnt/raid0/qwen2.5-coder-7b-instruct --port 11534

# Test
curl http://localhost:11534/api/version
```

---

## Hardware Configurations Tested

### AMD MI50 (Production)

- **GPU:** AMD Radeon Instinct MI50 32GB HBM2
- **Framework:** ROCm 6.2.4 + PyTorch 2.5.1
- **RAM:** 168GB (optimal for RAM-first loading)
- **Status:** ✅ Production ready
- **Performance:** 50-100 tok/s (Qwen 7B FP16)
- **Project:** `mi50-inference-server/`

### NVIDIA Tesla M40 (Embeddings)

- **GPU:** NVIDIA Tesla M40 12GB GDDR5
- **Framework:** ONNX Runtime (CPU mode)
- **Use Case:** RAG embeddings generation
- **Status:** ✅ Operational
- **Performance:** 3-8ms per embedding (CPU)

---

## Performance Highlights

### MI50 Optimization Journey

**Problem (Before):**
- VRAM: 90% idle (28GB on 32GB)
- Speed: 5-10 tokens/sec
- Latency: 20-30s for short responses
- Cause: Double allocation bug (PyTorch `device_map` + `low_cpu_mem_usage`)

**Solution:**
- RAM-first loading strategy
- Direct RAM → VRAM transfer
- Aggressive garbage collection
- GPU performance level = high

**Results (After):**
- VRAM: 50% idle (16GB on 32GB)
- Speed: 50-100 tokens/sec
- Latency: 2-5s for short responses
- 10x performance improvement

**See:** [mi50-inference-server/docs/08-performance-tuning.md](mi50-inference-server/docs/08-performance-tuning.md)

---

## Technologies

### Core ML Stack

- **PyTorch** 2.5.1+rocm6.2 - Deep learning framework
- **Transformers** 4.45 - HuggingFace model loading
- **ROCm** 6.2.4 - AMD GPU compute platform
- **ONNX Runtime** 1.23 - Optimized inference

### Web Frameworks

- **FastAPI** 0.122 - Modern async Python web framework
- **Uvicorn** - ASGI server
- **Flask** 3.0 - Dashboard framework

### Storage & Databases

- **ChromaDB** 1.3.5 - Vector database for RAG
- **JSON** - Chat persistence

### Deployment

- **Systemd** - Service management
- **Nginx** - Reverse proxy (optional)

---

## Supported Models

| Model Family | Sizes | Format | Status |
|--------------|-------|--------|--------|
| **Qwen 2.5 Coder** | 7B, 14B | FP16, GPTQ-Int4 | ✅ Recommended |
| **Gemma 2/3** | 4B, 9B | FP16 | ✅ Tested |
| **DeepSeek Coder** | 6.7B | FP16 | ✅ Tested |
| **Qwen 3** | 7B, 14B | FP16 | 🔄 Coming |

---

## License

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**

All projects in this repository are licensed under CC BY-NC-SA 4.0 unless otherwise specified.

**Commercial licensing available.** Contact: lele.sra@gmail.com

See individual project LICENSE.md files for details.

---

## Contributing

This is a personal experimental repository. However, feedback, bug reports, and suggestions are welcome!

**To report issues:**
1. Open a GitHub issue
2. Include hardware specs
3. Include relevant logs
4. Describe expected vs actual behavior

**For commercial support or consulting:**
- Email: lele.sra@gmail.com | info@axefactory.com
- Subject: "Local LLM Inference Lab - Support Request"

---

## Roadmap

### Completed ✅

- [x] MI50 backend with Ollama-compatible API
- [x] VRAM optimization (90% → 50%)
- [x] Web chat UI with token management
- [x] Monitoring dashboard
- [x] Systemd service integration
- [x] Comprehensive documentation (8 guides)
- [x] Tool calling support
- [x] Streaming responses

### Planned 🔄

- [ ] vLLM integration for MI50
- [ ] Multi-GPU support (MI50 + M40)
- [ ] Model quantization tools
- [ ] Fine-tuning pipeline
- [ ] Benchmark suite
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] WebGPU experiments

---

## Research Areas

### VRAM Optimization

**Findings:**
- RAM-first loading reduces VRAM waste 40%
- GC threshold 0.95 optimal for MI50
- KV cache allocation critical for throughput
- Single-model loading recommended (<32GB VRAM)

**Papers/Refs:**
- PyTorch Memory Allocator: [pytorch.org/docs/stable/notes/cuda.html](https://pytorch.org/docs/stable/notes/cuda.html)
- HBM2 Bandwidth Optimization: Internal benchmarks

### ROCm Performance

**Findings:**
- SDPA attention > eager (20% faster)
- Flash Attention 2 compatibility limited on MI50
- Performance level = high mandatory
- HIP_VISIBLE_DEVICES isolation reduces conflicts

---

## Benchmarks

### MI50 + Qwen 2.5 Coder 7B FP16

| Metric | Value | Config |
|--------|-------|--------|
| Throughput (greedy) | 100 tok/s | temp=0.0, short context |
| Throughput (sampling) | 70 tok/s | temp=0.7, top_p=0.9 |
| TTFT (first token) | 2-5s | prompt < 100 tokens |
| VRAM idle | 16GB / 32GB | 50% utilization |
| VRAM inference | 20-24GB / 32GB | 60-75% utilization |
| Max context | 32768 tokens | Model limit |

**Benchmark script:** `mi50-inference-server/backend/smoke_test.py`

---

## FAQ

**Q: Why AMD MI50 instead of NVIDIA?**
A: Cost-effective for research, 32GB VRAM, ROCm ecosystem maturity.

**Q: Can I use this on NVIDIA GPUs?**
A: Yes, with modifications (replace ROCm with CUDA, adjust memory management).

**Q: Why not use Ollama directly?**
A: More control over memory, custom optimizations, ROCm-specific tuning.

**Q: Production ready?**
A: MI50 Inference Server: Yes. Tested for months, stable, documented.

**Q: What about quantization?**
A: GPTQ-Int4 supported (download pre-quantized from HF). bitsandbytes not supported on ROCm.

---

## Related Projects

**By Raffaele Spezia:**

- **[ai-consciousness-research](https://github.com/RaffaeleSpezia/ai-consciousness-research)** - AI consciousness protocols (MAPS, NCIF, C.R.I.S.I.)
- **[functional-autonomy-manual](https://github.com/RaffaeleSpezia/functional-autonomy-manual)** - Framework for LLM functional autonomy

---

## Author

**Raffaele Spezia**
- GitHub: [@RaffaeleSpezia](https://github.com/RaffaeleSpezia)
- Email: lele.sra@gmail.com | info@axefactory.com
- Website: axefactory.com (TBD)

---

## Acknowledgments

- AMD for ROCm platform
- HuggingFace for Transformers library
- Qwen team (Alibaba Cloud) for excellent models
- FastAPI team for modern web framework
- Open source community

---

**Status:** Active Development 🚀
**Last Updated:** December 3, 2025
**Version:** 1.0
