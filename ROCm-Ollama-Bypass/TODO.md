# TODO / Backlog

## Immediate
- [ ] Implement multi-model registry (`ModelManager.list_models`) and expose `/api/tags` with lazy warm-loading.
- [ ] Persist chat history (e.g. optional SQLite/Redis backend instead of in-memory `SessionManager`).
- [ ] Provide sample prompts/templates per model family (Qwen, Llama, Mistral) to mirror Ollama's system prompts.
- [ ] Add a CLI flag/env var for `max_memory` to tune accelerate's VRAM reservation.

## Near Term Enhancements
- [ ] Optional RAG embedding on GPU when spare VRAM is available (configurable device selection).
- [ ] Authentication layer (API key or basic auth) for deployments exposed beyond localhost.
- [ ] Health/metrics endpoint (`/health`, `/metrics`) with GPU stats (utilisation, memory, temperature).
- [ ] Script/notebook showing how to download HuggingFace models into `/path/to/model` structure.

## Release Polish
- [ ] Record benchmarks (token throughput, latency) for MI50 + common 7B/8B models.
- [ ] Create HuggingFace Space or blog post demonstrating the workflow.
- [ ] Prepare GitHub Actions workflow (lint + tests) to run `pytest` and `smoke_test.py` with dummy model.
- [ ] Add LICENSE and CONTRIBUTING guidelines before making repo public.

## Stretch Ideas
- [ ] Hybrid deployment that routes CUDA GPUs (M30/M40) to bitsandbytes 4-bit loads while MI50 handles bf16 models.
- [ ] Provide Dockerfile / container instructions for ROCm images (requires attention to driver passthrough).
- [ ] Explore quantised model support on ROCm through experimental libraries (e.g. AutoGPTQ or bitsandbytes ROCm forks).

Tick off items as you complete them and cross-reference `DEVELOPMENT.md` for workflow details.
