# Development Notes

These notes help future you (or contributors) pick up work quickly.

## 1. Environment
- Python 3.10+ with ROCm-enabled PyTorch (2.1 or newer).
- Recommended layout: keep this repo under `/mnt/raid?/services/ROCm-Ollama-Bypass` and point caches/logs to the same disk for speed.
- Default virtualenv path is `.venv/` (see `start.sh`). Create it with:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- Environment variables used by the code: `HF_HOME`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `OLLAMA_FAKE_DEFAULT_MODEL`, `OLLAMA_FAKE_LOGDIR`, `OLLAMA_FAKE_LOGLEVEL`, `PYTHONPATH`.

## 2. Running the Service
- Preferred: `./start.sh "/abs/path/to/hf-model"` (handles env vars, logging, port checks). Use `PORT=11535` or similar to avoid conflicts.
- Manual fallback:
  ```bash
  source .venv/bin/activate
  export HF_HOME=...
  python app.py "/abs/path/to/model" --host 0.0.0.0 --port 11534 --log-level info
  ```
- Logs default to `logs/mi50_ollama.log` (JSON). Tweak `OLLAMA_FAKE_LOGDIR` if needed.
- Stop with `Ctrl+C` or `lsof -i:11534` → `kill <pid>`.

## 3. Testing & QA
- Unit/integration: `./run_tests.sh` (uses pytest + FastAPI TestClient).
- Smoke: `python smoke_test.py --host http://127.0.0.1:11534 --model "/abs/path/to/model"`.
- Manual curl (non-stream & stream) for sanity:
  ```bash
  curl -s http://127.0.0.1:11534/api/version
  curl -s -X POST http://127.0.0.1:11534/api/generate -H 'Content-Type: application/json' -d '{"prompt":"Ping"}'
  curl -N -X POST http://127.0.0.1:11534/api/generate -H 'Content-Type: application/json' -d '{"prompt":"Ping","stream":true}'
  ```

## 4. Release Checklist
- [ ] Run `./run_tests.sh` and smoke test against a real model.
- [ ] Update `README.md` and `TODO.md` with any new behaviour.
- [ ] Ensure `requirements.txt` reflects versions tested on ROCm.
- [ ] Regenerate systemd snippet with correct absolute paths before sharing.
- [ ] Tag release (optionally) and upload to GitHub / publish blog post / HF card.

## 5. Known Pitfalls
- ROCm emits `amdgpu.ids: No such file or directory` if `/etc/amdgpu/ids/amdgpu.ids` is missing. Safe to ignore or create an empty file.
- First request after boot is slow because weights are loaded into VRAM (check logs for `Loading ...`). Subsequent calls reuse the cached handle (`Model ... already loaded`).
- Keep MI50 service on a port distinct from official Ollama (`11534` by default) if both run simultaneously.
- RAG relies on sentence-transformers; CPU fallback is default but GPU can be forced by editing `rag_manager.py`.

## 6. Useful Commands
- `PORT=11536 ./start.sh` — quick way to spin up another instance.
- `OLLAMA_FAKE_DEFAULT_MODEL=/abs/path/model ./start.sh` — override default without CLI arg.
- `journalctl -u mi50_ollama.service -f` — follow logs when using systemd.
- `PYTHONPATH=$(pwd) pytest tests/test_app.py -k generate -q` — focus specific tests.

## 7. Next Steps Reference
See `TODO.md` for open tasks (multi-model registry, chat persistence, documentation polish, etc.).
