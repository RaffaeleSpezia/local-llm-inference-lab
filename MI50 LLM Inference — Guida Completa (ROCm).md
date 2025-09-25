

# MI50 LLM Inference — Guida Completa (ROCm)

> **Obiettivo:** mettere in funzione una **AMD Instinct MI50** per l’inferenza di modelli LLM in locale, con installazione ROCm, setup PyTorch/Transformers, esempi di codice, ottimizzazioni e troubleshooting.

## Indice

- [Panoramica](#panoramica)

- [Requisiti](#requisiti)

- [Preparazione BIOS e hardware](#preparazione-bios-e-hardware)

- [Installazione ROCm (driver + toolchain)](#installazione-rocm-driver--toolchain)

- [Setup ambiente Python (PyTorch ROCm + Transformers)](#setup-ambiente-python-pytorch-rocm--transformers)

- [Esempio 1 — Inferenza con Transformers su MI50](#esempio-1--inferenza-con-transformers-su-mi50)

- [Esempio 2 — llama.cpp con backend HIP/ROCm](#esempio-2--llamacpp-con-backend-hiprocm)

- [Ottimizzazioni](#ottimizzazioni)

- [Monitoraggio](#monitoraggio)

- [Troubleshooting](#troubleshooting)

- [FAQ rapide](#faq-rapide)

- [Licenza](#licenza)

---

## Panoramica

La **MI50** (Vega 20, 16–32 GB HBM2) è interessante per LLM locali grazie alla **molta VRAM** a costo contenuto. Con **ROCm** possiamo usare PyTorch/HF Transformers quasi come su CUDA, e/o **llama.cpp** via backend **HIP**.

> Risultato atteso: setup **stabile** e **abbastanza veloce** per modelli 7B–13B (fp16 o quantizzati), con esempi pronti all’uso.

---

## Requisiti

**Hardware**

- AMD Instinct **MI50** (consigliato 32 GB HBM2)

- PSU adeguato (≥ 800 W consigliati per server multi-GPU), **2×8-pin** PCIe

- Flusso d’aria elevato (dissipatore spesso passivo → servono ventole/chassis ben ventilato)

- Slot PCIe x16 fisico, board con supporto UEFI “Above 4G Decoding”

**Software**

- Linux (consigliato **Ubuntu 22.04 LTS** o simili)

- ROCm **compatibile con MI50** (serie 6.2–6.3; evitare versioni troppo nuove che deprecano gfx906)

- Python 3.10/3.11, `venv` o Conda

---

## Preparazione BIOS e hardware

1. **UEFI only (disabilita CSM/Legacy).**

2. **Above 4G Decoding: ON.** (fondamentale per GPU con molta VRAM)

3. **Resizable BAR**: se disponibile, abilitalo (non è strettamente necessario).

4. **Alimentazione**: collega **entrambi** gli 8-pin a linee PSU robuste (no adattatori scadenti).

5. **Raffreddamento**: assicurati di avere ventole che spingano aria attraverso il dissipatore della MI50.

Verifica da Linux:

```bash
lspci | grep -Ei "Vega 20|MI50|Radeon Pro VII"
```

---

## Installazione ROCm (driver + toolchain)

> Nota: usa una **release ROCm compatibile** con **gfx906** (MI50).

1. **Repo/installer AMD** (esempio indicativo; adatta la versione):

```bash
wget https://repo.radeon.com/amdgpu-install/22.40.3/ubuntu/jammy/amdgpu-install_22.40.3_all.deb
sudo dpkg -i amdgpu-install_22.40.3_all.deb
```

2. **Installazione componenti ROCm + driver kernel**

```bash
sudo amdgpu-install \
  --usecase=rocm,lrt,opencl,openclsdk,hip,hiplibsdk,dkms,mllib \
  --vulkan=amdvlk
```

3. **Riavvio** e **verifica**:

```bash
rocminfo      # la GPU deve apparire come agent gfx906
rocm-smi      # temperature, VRAM, power cap ecc.
lsmod | grep amdgpu
```

4. **Permessi device** (no root):

```bash
sudo usermod -a -G render,video $USER
# logout/login o riavvio
```

> Se in release ROCm troppo recenti la MI50 non viene rilevata, *solo come workaround* si può provare `export HSA_OVERRIDE_GFX_VERSION=9.0.6` (gfx906). Meglio restare su una versione nativamente compatibile.

---

## Setup ambiente Python (PyTorch ROCm + Transformers)

> Consigliato ambiente isolato: `venv` o Conda.

```bash
python3 -m venv ~/envs/rocm-llm
source ~/envs/rocm-llm/bin/activate
```

**PyTorch ROCm** (es. ROCm 6.3):

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

**Librerie LLM**

```bash
pip install transformers accelerate scipy huggingface_hub
```

**Smoke test**

```python
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Device name 0:", torch.cuda.get_device_name(0))
```

---

## Esempio 1 — Inferenza con Transformers su MI50

> Esempio con modello 7B fp16 (adatta al tuo preferito su HF Hub).

```python
# file: run_llm_rocm.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # cambia con il tuo modello

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,   # usa FP16
    device_map="auto"            # alloca sulla MI50
)

prompt = "Utente: Ciao, come stai?\nAssistente:"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))
```

Esecuzione:

```bash
python run_llm_rocm.py
```

---

## Esempio 2 — llama.cpp con backend HIP/ROCm

**Build**

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_HIP=ON
cmake --build build -j$(nproc)
```

**Inferenza** (modello GGUF quantizzato):

```bash
./build/bin/llama-cli \
  -m models/llama-2-7b.Q4_0.gguf \
  -p "Ciao, come stai?" \
  -n 128 \
  -ngl 100       # offload completo su GPU (se VRAM consente)
```

> Se non entra in VRAM, riduci `-ngl` o usa una quantizzazione più spinta (q4_k_m, q5_0, ecc.).

---

## Ottimizzazioni

- **Precisione**
  
  - Usa **FP16** come default per PyTorch.
  
  - Per modelli grandi → **quantizzazione 4/8-bit** (GGUF in llama.cpp; GPTQ/KV int4 dove possibile).

- **Throughput**
  
  - Servizi multi-utente: valuta **vLLM** (esistono fork per gfx906) per batching dinamico.
  
  - Multi-GPU: `Accelerate` può shardare i layer tra più MI50.

- **Memoria**
  
  - Evita batch > 1 in chat realtime.
  
  - Per contesti lunghi: preferisci quantizzazioni leggere + KV-cache attenta.

- **Stabilità**
  
  - Evita kill “bruschi” dei processi (reset bug). Chiudi pulito.
  
  - In rari casi di freeze durante trasferimenti pesanti: prova `HSA_ENABLE_SDMA=0`.

---

## Monitoraggio

Visione rapida:

```bash
watch -n 1 rocm-smi
```

Controlla **VRAM%**, **GPU%**, **Temp**, **Power** durante il `generate`.

In PyTorch:

```python
torch.cuda.reset_peak_memory_stats()
# ... inferenza ...
print(torch.cuda.max_memory_allocated() / 1024**3, "GB")
```

---

## Troubleshooting

**La GPU non appare in `rocminfo` / Torch → False**

- BIOS: **UEFI only**, **Above 4G Decoding ON**.

- Driver: reinstalla ROCm (versione compatibile gfx906).

- Permessi: aggiungi l’utente a `render,video` e riloggati.

- PCIe: usa slot x16 pieno; evita riser di bassa qualità.

**Segmentation fault / crash a VRAM alta**

- Probabile OOM/fragmentation. Riduci dimensione modello, quantizza, usa `device_map="auto"` per offload.

- Riavvia e rilancia subito l’inferenza (heap “pulito”).

- Valuta llama.cpp con GGUF 4-bit.

**Velocità molto bassa**

- Verifica che i tensori siano su `cuda` (ROCm) e non su CPU.

- Evita librerie quantizzazione solo-CUDA (bitsandbytes) → può forzare CPU.

- Scalda i kernel (la prima run è più lenta).

**Freeze / Reset GPU**

- Chiudi i processi correttamente; evita `SIGKILL`.

- In rari casi: `export HSA_ENABLE_SDMA=0` prima di avviare.

- Controlla alimentazione/temperature.

---

## FAQ rapide

**Posso usare Docker?**  
Sì, passando i device `/dev/dri/renderD*` al container e mantenendo lo stesso stack ROCm del sistema host. Per prestazioni e semplicità, **bare-metal** resta preferibile.

**BitsAndBytes funziona su AMD?**  
No (ad oggi è CUDA-only). Usa modelli già quantizzati (GGUF/GPTQ) o llama.cpp.

**Posso usare due GPU (MI50 + altra)?**  
Sì. Con `Accelerate device_map="auto"` puoi shardar layer tra GPU. Per throughput, esegui istanze separate o usa scheduler (vLLM/fork gfx906).

**Che modelli consigli per 16 GB?**  
7B **fp16** o 13B **int4**. Per 32 GB puoi spingerti a 13B fp16/8-bit e contesti più lunghi.

---

## Licenza

Rilascia questo README con la licenza del tuo repository (es. MIT/Apache-2.0). Le istruzioni qui presenti sono fornite “as-is”, senza garanzia.

---

### Appendice — Comandi utili

```bash
# Info ROCm
rocminfo | less
rocm-smi --showtemp --showuse --showmeminfo vram

# Env (usare solo se necessario)
export HSA_OVERRIDE_GFX_VERSION=9.0.6   # forza gfx906
export HSA_ENABLE_SDMA=0                # workaround freeze in rari casi

# PyTorch test veloce
python - <<'PY'
import torch
print(torch.cuda.get_device_name(0))
x = torch.randn(8192,8192, device='cuda', dtype=torch.float16)
y = x @ x.t()
print("OK:", y.mean().item())
PY
```

---









Parte pratica 





---

# 📁 Struttura suggerita del repo

```
mi50-llm/
├─ README.md
├─ install.sh
├─ requirements.txt
├─ .env.example
├─ config.yaml
├─ src/
│  ├─ server.py
│  ├─ run_llm_rocm.py
│  ├─ rocm_check.py
│  └─ utils.py
├─ systemd/
│  └─ mi50-llm.service
└─ scripts/
   ├─ start.sh
   └─ stop.sh
```

---

## 🧰 `install.sh` — installazione end-to-end (ROCm + ambiente Python)

```bash
#!/usr/bin/env bash
set -euo pipefail

# === Parametri personalizzabili ===
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VENV_DIR="${VENV_DIR:-$HOME/envs/rocm-llm}"
PIP_INDEX_ROCM="${PIP_INDEX_ROCM:-https://download.pytorch.org/whl/rocm6.3}"

echo "==> Preparazione ambiente virtuale: $VENV_DIR"
python${PYTHON_VERSION} -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

echo "==> Installo PyTorch ROCm"
pip install torch torchvision torchaudio --index-url "$PIP_INDEX_ROCM"

echo "==> Installo requirements del progetto"
pip install -r requirements.txt

echo "==> Check ROCm / PyTorch"
python src/rocm_check.py || {
  echo "ERRORE: controllo ROCm/PyTorch fallito"; exit 1;
}

echo "==> Installazione completata."
echo "Attiva l'ambiente con: source \"$VENV_DIR/bin/activate\""
echo "Imposta la configurazione in .env e config.yaml, poi avvia: scripts/start.sh"
```

---

## 📦 `requirements.txt`

```txt
transformers>=4.43
accelerate>=0.33
huggingface_hub>=0.24
fastapi>=0.112
uvicorn[standard]>=0.30
pydantic>=2.8
python-dotenv>=1.0
pyyaml>=6.0
scipy>=1.11
```

> Nota: bitsandbytes **non** incluso (CUDA-only). Tutto compatibile ROCm.

---

## ⚙️ `.env.example`

```ini
# Copia in .env e personalizza
MODEL_NAME=meta-llama/Llama-2-7b-hf
TORCH_DTYPE=fp16           # fp16 | bf16 | fp32 (fp16 consigliato)
MAX_NEW_TOKENS=256
TEMPERATURE=0.7
TOP_P=0.9
DO_SAMPLE=true

# Server
HOST=0.0.0.0
PORT=8000

# (workaround rari) abilita solo se serve
# HSA_ENABLE_SDMA=0
# HSA_OVERRIDE_GFX_VERSION=9.0.6
```

---

## 🗂 `config.yaml` — profili e limiti

```yaml
runtime:
  device_map: auto         # auto | cuda:0
  dtype: fp16              # fp16/bf16/fp32

model:
  name: "meta-llama/Llama-2-7b-hf"
  trust_remote_code: false
  low_cpu_mem_usage: true

generation:
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.9
  top_k: 0
  do_sample: true
  repetition_penalty: 1.05

limits:
  max_input_tokens: 4096
  timeout_s: 120
```

---

## 🧪 `src/rocm_check.py` — smoke test ROCm/Torch

```python
import os, torch

def main():
    print("Torch:", torch.__version__)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm non disponibile in Torch")
    n = torch.cuda.device_count()
    print("GPU count:", n)
    for i in range(n):
        print(f" - [{i}] {torch.cuda.get_device_name(i)}")
    # test rapido matmul su GPU 0
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    y = x @ x.T
    print("OK, mean:", float(y.mean().cpu()))
    print("ROCm check: PASS")
if __name__ == "__main__":
    main()
```

---

## 🧩 `src/utils.py` — helper comuni

```python
import os, yaml, torch
from dataclasses import dataclass
from typing import Any, Dict

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def env_or(default: str, key: str) -> str:
    return os.getenv(key, default)

def resolve_dtype(s: str) -> torch.dtype:
    return DTYPE_MAP.get(s.lower(), torch.float16)
```

---

## 🧪 `src/run_llm_rocm.py` — esempio standalone (CLI)

```python
import os, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import load_yaml, resolve_dtype

CFG = load_yaml(os.getenv("CFG", "config.yaml"))
ENV_MODEL = os.getenv("MODEL_NAME", CFG["model"]["name"])

def main():
    tokenizer = AutoTokenizer.from_pretrained(ENV_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        ENV_MODEL,
        torch_dtype=resolve_dtype(os.getenv("TORCH_DTYPE", CFG["runtime"]["dtype"])),
        device_map=CFG["runtime"]["device_map"],
        trust_remote_code=CFG["model"].get("trust_remote_code", False),
        low_cpu_mem_usage=CFG["model"].get("low_cpu_mem_usage", True),
    )
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Utente: Ciao!\nAssistente:"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    gen_cfg = CFG["generation"]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", gen_cfg["max_new_tokens"])),
            temperature=float(os.getenv("TEMPERATURE", gen_cfg["temperature"])),
            top_p=float(os.getenv("TOP_P", gen_cfg["top_p"])),
            top_k=int(gen_cfg["top_k"]),
            do_sample=(os.getenv("DO_SAMPLE", str(gen_cfg["do_sample"])).lower() == "true"),
            repetition_penalty=float(gen_cfg["repetition_penalty"]),
        )
    print(tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
```

---

## 🌐 `src/server.py` — FastAPI (REST + streaming)

```python
import os, asyncio, torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from dotenv import load_dotenv
from utils import load_yaml, resolve_dtype

load_dotenv(".env")
CFG = load_yaml(os.getenv("CFG", "config.yaml"))

MODEL_NAME = os.getenv("MODEL_NAME", CFG["model"]["name"])
DTYPE = resolve_dtype(os.getenv("TORCH_DTYPE", CFG["runtime"]["dtype"]))
DEVICE_MAP = CFG["runtime"]["device_map"]

app = FastAPI(title="MI50 LLM Server", version="1.0.0")

# Lazy-load globale
_tokenizer = None
_model = None

def get_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            device_map=DEVICE_MAP,
            trust_remote_code=CFG["model"].get("trust_remote_code", False),
            low_cpu_mem_usage=CFG["model"].get("low_cpu_mem_usage", True),
        )
    return _tokenizer, _model

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    do_sample: bool | None = None
    stream: bool = False

@app.get("/health")
def health():
    try:
        name = torch.cuda.get_device_name(0)
        return {"status": "ok", "device": name}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "err", "detail": str(e)})

@app.post("/generate")
def generate(req: GenerateRequest):
    tokenizer, model = get_model()
    gen_cfg = CFG["generation"]

    max_new_tokens = req.max_new_tokens or int(os.getenv("MAX_NEW_TOKENS", gen_cfg["max_new_tokens"]))
    temperature = req.temperature or float(os.getenv("TEMPERATURE", gen_cfg["temperature"]))
    top_p = req.top_p or float(os.getenv("TOP_P", gen_cfg["top_p"]))
    do_sample = req.do_sample if req.do_sample is not None else (os.getenv("DO_SAMPLE", str(gen_cfg["do_sample"])).lower() == "true")

    inputs = tokenizer(req.prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    if not req.stream:
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=float(gen_cfg["repetition_penalty"]),
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        return {"text": text}

    # Streaming (token-by-token)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        repetition_penalty=float(gen_cfg["repetition_penalty"]),
        streamer=streamer,
    )

    def token_generator():
        with torch.inference_mode():
            thread = asyncio.get_event_loop().run_in_executor(None, model.generate, **gen_kwargs)
            for token in streamer:
                yield token
            # ensure completion
            asyncio.ensure_future(thread)

    return StreamingResponse(token_generator(), media_type="text/plain")
```

---

## 🧯 `systemd/mi50-llm.service` — servizio di sistema

```ini
[Unit]
Description=MI50 LLM FastAPI Server
After=network.target

[Service]
Type=simple
Environment=PYTHONUNBUFFERED=1
# opzionali mitigazioni
# Environment=HSA_ENABLE_SDMA=0
# Environment=HSA_OVERRIDE_GFX_VERSION=9.0.6
WorkingDirectory=/opt/mi50-llm
ExecStart=/bin/bash -lc 'source $HOME/envs/rocm-llm/bin/activate && uvicorn src.server:app --host 0.0.0.0 --port 8000'
Restart=on-failure
User=YOURUSER
Group=YOURUSER

[Install]
WantedBy=multi-user.target
```

> Copia il progetto in `/opt/mi50-llm`, sistema `User`, poi:
> 
> ```bash
> sudo cp systemd/mi50-llm.service /etc/systemd/system/
> sudo systemctl daemon-reload
> sudo systemctl enable --now mi50-llm
> sudo systemctl status mi50-llm
> ```

---

## ▶️ `scripts/start.sh` & `scripts/stop.sh`

```bash
# scripts/start.sh
#!/usr/bin/env bash
set -e
source "${VENV_DIR:-$HOME/envs/rocm-llm}/bin/activate"
export CFG="${CFG:-config.yaml}"
export $(grep -v '^#' .env | xargs -d '\n') || true
uvicorn src.server:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
```

```bash
# scripts/stop.sh
#!/usr/bin/env bash
pkill -f "uvicorn src.server:app" || true
```

> Ricorda `chmod +x scripts/*.sh install.sh`.

---

## 🧪 Prove rapide

- **Check ROCm/Torch**
  
  ```bash
  source ~/envs/rocm-llm/bin/activate
  python src/rocm_check.py
  ```

- **CLI singola**
  
  ```bash
  python src/run_llm_rocm.py "Utente: dimmi una curiosità sulla MI50.\nAssistente:"
  ```

- **Server**
  
  ```bash
  scripts/start.sh
  # in altro terminale:
  curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Utente: scrivi un haiku sulla HBM2.\nAssistente:"}'
  ```

- **Streaming**
  
  ```bash
  curl -N -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Utente: crea una risposta lunga.\nAssistente:","stream":true}'
  ```

---

Se vuoi, aggiungo anche un **Dockerfile** con pass-through `/dev/dri/renderD*` (host ROCm), ma per prestazioni e semplicità ti consiglio bare-metal. Dimmi se lo vuoi e lo preparo. 💜
