# Performance Tuning - Ottimizzazione Avanzata

**Versione:** 1.0
**Data:** 3 Dicembre 2025
**Autore:** Raffaele Spezia

---

## Indice

1. [VRAM Optimization](#vram-optimization)
2. [Inference Speed](#inference-speed)
3. [Model Selection](#model-selection)
4. [Prompt Engineering](#prompt-engineering)
5. [System Tuning](#system-tuning)
6. [Benchmarking](#benchmarking)

---

## VRAM Optimization

### RAM-First Loading Strategy

**Principio:** Server con RAM abbondante (168GB) può usare RAM come buffer per ottimizzare VRAM.

**Implementazione:** (già applicata in model_manager.py)

```python
# ❌ APPROACH VECCHIO (VRAM 90%)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",           # Causa double allocation
    low_cpu_mem_usage=True       # Chunk loading → buffer VRAM
)

# ✅ APPROACH OTTIMIZZATO (VRAM 50%)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16   # No device_map, no low_cpu_mem_usage
)
model = model.to("cuda:0")      # Transfer diretto RAM → VRAM
torch.cuda.empty_cache()        # Cleanup buffer temporanei
```

**Risultati:**
- VRAM idle: 50% invece di 90%
- 16GB liberi per KV cache (era 4GB)
- Velocità: 50-100 tok/s (era 5-10 tok/s)
- Latenza: 2-5s (era 20-30s)

**Quando applicabile:**
- Server con RAM > 64GB
- Modelli FP16 fino a 14B (~28GB)
- Single-model loading (non multi-model concurrent)

**Quando NON applicabile:**
- Server con RAM < 32GB → usa `low_cpu_mem_usage=True`
- Multi-GPU setup → usa `device_map="auto"`

---

### Garbage Collection Tuning

**Configurazione allocator PyTorch:**

```bash
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.95"
```

**Parametri:**

| Parametro | Valore | Effetto |
|-----------|--------|---------|
| `max_split_size_mb` | 128 | Limite dimensione frammenti VRAM (riduce frammentazione) |
| `garbage_collection_threshold` | 0.95 | GC aggressivo al 95% utilizzo VRAM |

**Alternative tuning:**

```bash
# Conservative (meno GC, più frammentazione)
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.98"

# Aggressive (più GC, meno frammentazione, possibile overhead)
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:64,garbage_collection_threshold:0.90"
```

**Test quale funziona meglio:**

```bash
# Test 1: Conservative
export PYTORCH_HIP_ALLOC_CONF="..."
./start.sh
# → Run 10 generazioni
# → Misura VRAM waste (reserved - allocated)

# Test 2: Aggressive
# → Repeat
# → Compare waste
```

---

### Model Unloading

**Automatic unload prima di caricare nuovo modello:** (già implementato)

```python
# In model_manager.py load_model()
if self._models:  # Previous models exist
    for old_model in list(self._models.keys()):
        handle = self._models.pop(old_model)
        del handle.model
        del handle.tokenizer
    torch.cuda.empty_cache()
```

**Manual unload API:**

```bash
# Unload current model
curl -X POST http://192.168.1.155:11534/api/delete \
  -H 'Content-Type: application/json' \
  -d '{"name":"/mnt/raid0/qwen2.5-coder-7b-instruct"}'

# Check VRAM freed
curl http://192.168.1.155:11534/debug/memory | jq
# allocated_gb dovrebbe essere ~0.5GB (solo overhead)
```

---

## Inference Speed

### GPU Performance Level

**CRITICO:** Default `auto` non scala clock oltre 925MHz.

```bash
# Set high performance
sudo rocm-smi --setperflevel high

# Verify
rocm-smi | grep Perf
# Output: Perf = high
```

**Automazione al boot:**

```bash
# Create systemd service
sudo tee /etc/systemd/system/rocm-performance.service << 'EOF'
[Unit]
Description=Set ROCm GPU Performance High
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/opt/rocm/bin/rocm-smi --setperflevel high
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable rocm-performance.service
sudo systemctl start rocm-performance.service
```

---

### Generation Parameters

**Fast Greedy (default):**
```json
{
  "temperature": 0.0,
  "top_k": 0,
  "top_p": 1.0,
  "max_new_tokens": 128
}
```
- Velocità: massima (~100 tok/s)
- Qualità: deterministica
- Uso: code generation, factual answers

**Balanced Creative:**
```json
{
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.9,
  "max_new_tokens": 512
}
```
- Velocità: media (~70 tok/s)
- Qualità: creativa ma controllata
- Uso: chat, spiegazioni

**Highly Creative:**
```json
{
  "temperature": 1.2,
  "top_k": 100,
  "top_p": 0.95,
  "max_new_tokens": 1024
}
```
- Velocità: più lenta (~50 tok/s)
- Qualità: molto creativa, può deviare
- Uso: storytelling, brainstorming

**Tradeoff temperatura vs velocità:**

| Temperature | Tok/s (Qwen 7B) | Overhead |
|-------------|-----------------|----------|
| 0.0 (greedy) | 100 | 0% |
| 0.5 | 90 | +11% |
| 0.7 | 80 | +25% |
| 1.0 | 70 | +43% |
| 1.5 | 60 | +67% |

**Overhead dovuto a:** sampling, sorting, multinomial distribution.

---

### Streaming Chunk Size

**Config:**
```bash
export OLLAMA_FAKE_STREAM_CHARS="160"
```

**Effetti:**

| Chunk Size | Latency Percepita | Overhead JSON | Throughput |
|------------|-------------------|---------------|------------|
| 0 (token-by-token) | Minima | Alto (10x) | Basso |
| 40 | Bassa | Medio | Medio |
| 160 (default) | Media | Basso | Alto |
| 320 | Alta | Minimo | Massimo |

**Recommendation:**
- UI interattiva: 40-80 (latency bassa)
- Batch processing: 320+ (throughput alto)
- Default 160: buon compromesso

---

### Attention Implementation

**Config:**
```bash
export OLLAMA_FAKE_ATTN_IMPL="sdpa"
```

**Options:**

| Impl | Speed | Memory | Compatibility |
|------|-------|--------|---------------|
| `sdpa` | ⭐⭐⭐ | ⭐⭐⭐ | ROCm 5.x+ |
| `flash_attention_2` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ROCm 6.1+, non tutti modelli |
| `eager` | ⭐⭐ | ⭐ | Tutti |

**Test flash_attention_2:**

```bash
export OLLAMA_FAKE_ATTN_IMPL="flash_attention_2"
./start.sh

# Se errore → fallback a sdpa
# Check log: "Falling back to sdpa attention"
```

**Benchmark:**

```bash
# Benchmark sdpa vs flash_attention_2
for impl in sdpa flash_attention_2; do
  export OLLAMA_FAKE_ATTN_IMPL=$impl
  pkill pt_main_t
  ./start.sh &
  sleep 60  # Wait load

  # Measure
  time curl -s -X POST http://127.0.0.1:11534/api/generate \
    -d '{"prompt":"Write 500 word essay on AI","options":{"max_new_tokens":512}}' \
    | jq -r '.eval_count, .total_duration'
done
```

---

## Model Selection

### Model Size vs Speed

**Benchmark (MI50, FP16):**

| Modello | VRAM | Tok/s | Quality | Use Case |
|---------|------|-------|---------|----------|
| **Gemma3 4B** | 8GB | 120 | ⭐⭐⭐ | Fast chat, simple tasks |
| **DeepSeek 6.7B** | 13GB | 90 | ⭐⭐⭐⭐ | Code generation |
| **Qwen 7B** | 14GB | 100 | ⭐⭐⭐⭐ | General purpose ⭐ |
| **Gemma 2 9B** | 18GB | 70 | ⭐⭐⭐⭐ | Balanced |
| **Qwen 14B** | 28GB | 55 | ⭐⭐⭐⭐⭐ | Complex reasoning |
| **Qwen 14B GPTQ-Int4** | 14GB | 75 | ⭐⭐⭐⭐ | Speed + Quality ⭐ |

**Recommendation:**
- **Default:** Qwen 7B (best speed/quality)
- **Speed critical:** Gemma3 4B
- **Quality critical:** Qwen 14B GPTQ-Int4 (not FP16)

---

### Quantization

**GPTQ Int4:**
- Size: -50% (28GB → 14GB)
- Speed: +20% (VRAM freed per KV cache)
- Quality: -5% (acceptable trade-off)

**Download GPTQ model:**

```bash
cd /mnt/raid0
python3 << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4",
    local_dir="/mnt/raid0/qwen2.5-coder-14b-gptq-int4",
    local_dir_use_symlinks=False
)
EOF
```

**Test GPTQ vs FP16:**

```bash
# Load GPTQ
python app.py /mnt/raid0/qwen2.5-coder-14b-gptq-int4 --port 11534 &
# → Check VRAM: ~14GB
# → Benchmark tok/s

# Compare FP16
python app.py /mnt/raid0/qwen2.5-coder-14b-instruct --port 11534 &
# → Check VRAM: ~28GB
# → Benchmark tok/s
```

---

## Prompt Engineering

### Context Length Optimization

**Principio:** Prompt più corto = inferenza più veloce.

**Bad prompt (verbose):**
```
I would like you to help me write a Python function that can reverse a string. Please make sure to include comments explaining each step of the process, and also provide an example of how to use the function with a test case.
```
→ 50 tokens

**Good prompt (concise):**
```
Write a Python function to reverse a string with comments and example.
```
→ 12 tokens

**Speedup:** ~4x prompt eval time.

---

### System Message Tuning

**Lightweight system:**
```
You are a helpful assistant.
```
→ 5 tokens

**Heavy system (unnecessary):**
```
You are an advanced AI assistant powered by state-of-the-art large language models, designed to provide comprehensive, accurate, and helpful responses across a wide range of topics including but not limited to science, technology, history, mathematics...
```
→ 60+ tokens

**Impact:** System message processato ad ogni request → overhead cumulativo.

---

### Stop Sequences

**Config:**
```json
{
  "prompt": "Write Python function:",
  "options": {
    "stop": ["\n\n", "```", "def "]
  }
}
```

**Effect:** LLM stops quando incontra stop sequence → evita over-generation.

**Use case:**
- Code: `["```", "\n\n\n"]`
- Single answer: `["\n", "Q:"]`
- Dialog: `["User:", "Assistant:"]`

---

## System Tuning

### Kernel Parameters

**Transparent Huge Pages (THP):**

```bash
# Check current
cat /sys/kernel/mm/transparent_hugepage/enabled
# [always] madvise never

# Set madvise (recommended)
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

# Permanent (add to /etc/rc.local)
```

**Effect:** Reduce TLB misses → +5-10% throughput.

---

### CPU Governor

**Set performance mode:**

```bash
# Check current
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# powersave

# Set performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Verify
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# performance
```

**Effect:** CPU non throttles → tokenizer più veloce.

---

### Swap Configuration

**Disable swap (se RAM > 128GB):**

```bash
sudo swapoff -a

# Check
free -h | grep Swap
# Swap: 0B
```

**Effect:** Evita swap latency se VRAM spills.

**WARNING:** Solo se RAM abbondante. Con RAM < 64GB, mantieni swap.

---

## Benchmarking

### Tokens/Second Test

**Script:**

```bash
#!/bin/bash

BACKEND="http://127.0.0.1:11534"
PROMPT="Write a detailed essay about artificial intelligence and its impact on society"
TOKENS=512

for temp in 0.0 0.5 0.7 1.0; do
  echo "Testing temperature $temp..."

  start=$(date +%s%N)
  response=$(curl -s -X POST $BACKEND/api/generate \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"$PROMPT\",\"options\":{\"temperature\":$temp,\"max_new_tokens\":$TOKENS}}")

  end=$(date +%s%N)
  duration=$(( (end - start) / 1000000 ))  # ms
  eval_count=$(echo "$response" | jq -r '.eval_count')
  tok_per_sec=$(echo "scale=2; $eval_count / ($duration / 1000)" | bc)

  echo "  Duration: ${duration}ms"
  echo "  Tokens: $eval_count"
  echo "  Tok/s: $tok_per_sec"
  echo ""
done
```

---

### Latency Test

**First-token latency (TTFT):**

```python
import time
import requests

def measure_ttft(prompt):
    url = "http://192.168.1.155:11534/api/generate"
    data = {"prompt": prompt, "stream": True, "options": {"max_new_tokens": 1}}

    start = time.time()
    response = requests.post(url, json=data, stream=True)

    for line in response.iter_lines():
        if line:
            ttft = (time.time() - start) * 1000  # ms
            print(f"TTFT: {ttft:.2f}ms")
            break

measure_ttft("Hello")
```

**Target TTFT:**
- Short prompt (<100 tok): < 500ms
- Medium prompt (100-500 tok): < 2s
- Long prompt (500-2000 tok): < 5s

---

### VRAM Efficiency

**Measure waste:**

```bash
curl -s http://127.0.0.1:11534/debug/memory | jq '{
  allocated: .allocated_gb,
  reserved: .reserved_gb,
  waste: (.reserved_gb - .allocated_gb),
  efficiency: ((.allocated_gb / .reserved_gb) * 100 | floor)
}'
```

**Target:**
- Waste < 2GB
- Efficiency > 90%

---

## Performance Checklist

**Before production:**

- [ ] GPU performance level = high
- [ ] VRAM idle < 55%
- [ ] VRAM waste < 2GB
- [ ] Tok/s > 50 (Qwen 7B)
- [ ] TTFT < 2s (short prompt)
- [ ] Streaming chunk size = 160
- [ ] Attention impl = sdpa
- [ ] System message < 20 tokens
- [ ] Stop sequences configured
- [ ] No swap usage
- [ ] Logs on ramdisk (/dev/shm)

---

**Licenza:** CC BY-NC-SA 4.0
**Autore:** Raffaele Spezia
**Repository:** https://github.com/RaffaeleSpezia/local-llm-inference-lab
