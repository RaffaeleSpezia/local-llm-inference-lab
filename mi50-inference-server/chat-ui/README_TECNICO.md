# MI50 Chat UI - Documentazione Tecnica

## Architettura Sistema

```
┌─────────────────┐
│   Browser       │
│  (port 13010)   │
└────────┬────────┘
         │ HTTP/WebSocket
         ▼
┌─────────────────┐
│   Chat UI       │
│  FastAPI        │
│  (port 13010)   │
└────────┬────────┘
         │ HTTP POST /api/generate
         ▼
┌─────────────────┐
│  Backend MI50   │
│  (port 11534)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   GPU MI50      │
│   32GB VRAM     │
└─────────────────┘
```

## Stack Tecnologico

### Frontend
- **HTML5** + **JavaScript vanilla** (no framework)
- **CSS3** (dark theme, responsive)
- **Fetch API** per streaming

### Backend Chat UI
- **FastAPI** 0.110.2
- **Uvicorn** 0.30.1
- **httpx** 0.27.0 (client async)
- **Python** 3.11

### Backend MI50
- **PyTorch** con ROCm
- **Transformers** (HuggingFace)
- **FastAPI**

---

## Dettaglio Componenti

### 1. storage.py

**Responsabilità:** Gestione persistenza chat su JSON file

**Classi principali:**

```python
class ChatStorage:
    def __init__(self, path: Path, default_model: str)
    
    # CRUD chat
    def create_chat(title, model, system_message) -> dict
    def get_chat(chat_id: str) -> dict
    def list_chats() -> list
    def update_title(chat_id, title)
    def update_system_message(chat_id, system_message)
    
    # Messaggi
    def append_message(chat_id, role, content)
    def to_openai_messages(chat_id) -> list
```

**Formato chat:**
```json
{
  "chats": [
    {
      "id": "uuid",
      "title": "Sessione gemma-7b",
      "model": "/mnt/raid0/gemma-7b",
      "system_message": "Sei un assistente...",
      "tags": ["Python", "Codice"],
      "created_at": "2025-11-19T...",
      "updated_at": "2025-11-19T...",
      "messages": [
        {"role": "user", "content": "...", "ts": "..."},
        {"role": "assistant", "content": "...", "ts": "..."}
      ]
    }
  ]
}
```

### 2. prompt_formatter.py

**Responsabilità:** Costruire prompt con tag corretti per ogni modello

**Funzioni principali:**

```python
def format_gemma3_prompt(messages: List[Dict]) -> str
def format_qwen_prompt(messages: List[Dict]) -> str
def detect_model_family(model_path: str) -> str
def format_prompt_for_model(model_path: str, messages: List[Dict]) -> str
```

**Esempio output Gemma:**
```
<start_of_turn>user
System: Sei un assistente

Domanda utente<end_of_turn>
<start_of_turn>model
```

**Esempio output Qwen:**
```
<|im_start|>system
Sei un assistente<|im_end|>
<|im_start|>user
Domanda<|im_end|>
<|im_start|>assistant
```

### 3. main.py

**Responsabilità:** API server e orchestrazione

**Endpoints:**

| Metodo | Path | Descrizione |
|--------|------|-------------|
| GET | `/` | Serve index.html |
| GET | `/api/chats` | Lista tutte le chat |
| POST | `/api/chats` | Crea nuova chat |
| GET | `/api/chats/{id}` | Dettagli chat |
| POST | `/api/chats/{id}/send` | Invia messaggio (streaming) |
| POST | `/api/chats/{id}/rename` | Rinomina chat |
| POST | `/api/chats/{id}/system` | Aggiorna system message |

**Flusso send message:**

```python
@app.post("/api/chats/{chat_id}/send")
async def send_message(chat_id, payload):
    # 1. Salva messaggio user
    storage.append_message(chat_id, "user", payload.prompt)
    
    # 2. Costruisci prompt formattato
    messages = storage.to_openai_messages(chat_id)
    formatted_prompt = format_prompt_for_model(chat["model"], messages)
    
    # 3. Prepara request per backend
    request_payload = {
        "model": chat["model"],
        "prompt": formatted_prompt,
        "stream": True,
        "options": {
            "temperature": ...,
            "top_p": ...,
            # ...
        }
    }
    
    # 4. Stream response
    async def stream():
        async with client.stream("POST", f"{BACKEND}/api/generate", json=request_payload):
            for line in resp.aiter_lines():
                data = json.loads(line)
                yield json.dumps({"response": data["response"]})
        
        # 5. Salva risposta assistant
        storage.append_message(chat_id, "assistant", full_text)
    
    return StreamingResponse(stream(), media_type="application/x-ndjson")
```

### 4. index.html

**Responsabilità:** UI interattiva

**Componenti UI:**

```
┌──────────────────────────────────────┐
│  Sidebar (300px)                     │
│  ┌────────────────────────────────┐  │
│  │ Selezione Modello              │  │
│  ├────────────────────────────────┤  │
│  │ Parametri (Temp/P/K/Tokens)    │  │
│  ├────────────────────────────────┤  │
│  │ Legenda Tag Modello            │  │
│  ├────────────────────────────────┤  │
│  │ Lista Sessioni Chat (scroll)   │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│  Chat Area                           │
│  ┌────────────────────────────────┐  │
│  │ Header (titolo + modello)      │  │
│  ├────────────────────────────────┤  │
│  │ Messaggi (scrollabile)         │  │
│  │  ┌──────────────────────┐      │  │
│  │  │ User bubble          │      │  │
│  │  └──────────────────────┘      │  │
│  │      ┌──────────────────────┐  │  │
│  │      │ Assistant bubble     │  │  │
│  │      └──────────────────────┘  │  │
│  ├────────────────────────────────┤  │
│  │ Composer (textarea + buttons)  │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

**Funzioni JavaScript chiave:**

```javascript
// Gestione chat
function loadChats()
function selectChat(chatId)
function createChat()

// Messaggi
function streamResponse(prompt)
function addBubble(role, content)

// Parametri
function getGenerationParams()
```

**Streaming NDJSON:**
```javascript
fetch('/api/chats/{id}/send', {
  method: 'POST',
  body: JSON.stringify({prompt, options})
})
.then(res => {
  const reader = res.body.getReader()
  // Leggi chunk per chunk
  reader.read().then(processChunk)
})

function processChunk({value, done}) {
  const lines = decoder.decode(value).split('\n')
  lines.forEach(line => {
    const data = JSON.parse(line)
    fullText += data.response
    updateBubble(fullText)
  })
}
```

---

## Gestione Memoria GPU

### Problema Identificato

PyTorch su ROCm pre-alloca VRAM in modo aggressivo:

```
Allocata:  14.5 GB (modello)
Riservata: 31.5 GB (buffer futuro)
Libera:    0.25 GB → OOM!
```

Durante generazione serve allocare:
- Input tensors: ~500 MB
- KV cache: 1-3 GB (cresce con i token)
- Attention buffers: ~600 MB

Con solo 250MB liberi → **Out Of Memory**

### Soluzione Implementata

**File:** `mi50_come_ollama/model_manager.py`

**Fix 1 - Dopo ogni generazione (linea ~285):**
```python
yield ""  # signal completion

# Libera memoria GPU dopo generazione
del inputs
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

**Fix 2 - Dopo caricamento modello (linea ~172):**
```python
model.eval()

# Forza garbage collection dopo eval
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    LOGGER.info("Freed reserved VRAM after model.eval()")
```

**Risultato:**
```
Allocata:  15.0 GB
Riservata: 17.8 GB
Libera:    13.8 GB ✅
```

### Configurazione Allocatore

In `start.sh`:
```bash
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.95,expandable_segments:True"
```

**Parametri:**
- `max_split_size_mb:128` - Blocchi max 128MB (riduce frammentazione)
- `garbage_collection_threshold:0.95` - GC quando >95% pieno
- `expandable_segments:True` - Segmenti espandibili invece di fissi

---

## Troubleshooting

### Chat UI non parte

**Sintomi:** Errore porta occupata o processo non risponde

**Diagnosi:**
```bash
lsof -iTCP:13010 -sTCP:LISTEN
ps aux | grep uvicorn
```

**Soluzione:**
```bash
pkill -f "uvicorn.*13010"
cd ~/mi50_stack/mi50_chat_ui
./start_chat_ui.sh
```

### Backend non risponde

**Sintomi:** Timeout, errore connessione

**Diagnosi:**
```bash
lsof -iTCP:11534 -sTCP:LISTEN
ps aux | grep pt_main_t
curl http://127.0.0.1:11534/api/tags
```

**Soluzione:**
```bash
pkill -f pt_main_t
./start_mi50_ai_server.sh qwen2.5-coder-7b-instruct
```

### Out Of Memory

**Sintomi:**
```
torch.OutOfMemoryError: HIP out of memory
```

**Diagnosi:**
```bash
curl http://127.0.0.1:11534/debug/memory
rocm-smi
```

**Verifica fix applicata:**
```bash
grep -A 3 "Libera memoria GPU" ~/mi50_stack/mi50_come_ollama/model_manager.py
```

Se non presente, ripristina da backup e riapplica fix.

### Modello genera loop infiniti

**Sintomi:** Risposta ripetuta o inventata

**Causa:** Prompt mal formattato per il modello

**Verifica:**
1. Controlla che `prompt_formatter.py` sia presente
2. Verifica che `main.py` usi `format_prompt_for_model()`
3. Controlla console browser per errori JavaScript

**Debug prompt inviato:**
Nel browser, apri console (F12):
```javascript
// I log mostrano il prompt formattato
[streamResponse] Invio a chat: ...
```

Oppure guarda i log backend:
```bash
tail -f /dev/shm/mi50_ollama_logs/mi50_ollama.log | grep -A 10 "Prompt"
```

### Sessioni chat non visibili

**Sintomi:** Lista vuota o non scorrevole

**Verifica:**
1. Check file JSON: `cat ~/mi50_stack/mi50_chat_ui/chat_state.json`
2. Console browser per errori JavaScript
3. Network tab per verificare risposta `/api/chats`

**Fix:**
Se JSON corrotto:
```bash
echo '{"chats":[]}' > ~/mi50_stack/mi50_chat_ui/chat_state.json
```

---

## Performance Tuning

### Parametri Temperature/Top P/K

**Temperature (0-2):**
- `0.0-0.3`: Deterministico, preciso (codice, FAQ)
- `0.7-1.0`: Bilanciato (conversazione)
- `1.0-2.0`: Creativo, variabile (brainstorming)

**Top P (0-1):**
- `0.9-0.95`: Standard (buon bilanciamento)
- `0.95-1.0`: Più varietà
- `0.7-0.9`: Più focalizzato

**Top K (0-100):**
- `0`: Disabilitato (usa solo Top P)
- `40-50`: Standard
- `80-100`: Più varietà

**Max Tokens:**
- Codice: 1024-2048
- Conversazione: 512-1024
- Lunghi documenti: 4096-8192

### Ottimizzazione VRAM

**Per modelli grandi (14B):**
```bash
# Riduci max context nel backend
export OLLAMA_FAKE_MAX_PROMPT_TOKENS=2048

# Riduci max_tokens nella chat_ui
Max Tokens slider: 1024 invece di 2048
```

**Monitoraggio continuo:**
```bash
watch -n 1 'curl -s http://127.0.0.1:11534/debug/memory | grep -E "allocata|libera"'
```

---

## Backup e Restore

### Backup Completo

```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=~/backups/mi50_chat_ui

mkdir -p $BACKUP_DIR

# Backup codice
tar -czf $BACKUP_DIR/chat_ui_code_$DATE.tar.gz \
    ~/mi50_stack/mi50_chat_ui/app \
    ~/mi50_stack/mi50_chat_ui/static \
    ~/mi50_stack/mi50_chat_ui/*.sh

# Backup conversazioni
cp ~/mi50_stack/mi50_chat_ui/chat_state.json \
   $BACKUP_DIR/chat_state_$DATE.json

# Backup model_manager modificato
cp ~/mi50_stack/mi50_come_ollama/model_manager.py \
   $BACKUP_DIR/model_manager_$DATE.py

echo "✅ Backup completato in $BACKUP_DIR"
```

### Restore

```bash
# Restore chat_state
cp ~/backups/mi50_chat_ui/chat_state_YYYYMMDD.json \
   ~/mi50_stack/mi50_chat_ui/chat_state.json

# Restore codice
tar -xzf ~/backups/mi50_chat_ui/chat_ui_code_YYYYMMDD.tar.gz -C ~

# Riavvia servizi
pkill -f "uvicorn.*13010"
cd ~/mi50_stack/mi50_chat_ui && ./start_chat_ui.sh
```

---

## Log e Monitoring

### Posizioni Log

| Servizio | Path |
|----------|------|
| Chat UI | `~/mi50_stack/mi50_chat_ui/chat_ui.log` |
| Backend MI50 | `/dev/shm/mi50_ollama_logs/mi50_ollama.log` |
| Startup backend | `/tmp/chat_ui_*.log` |

### Monitoring VRAM

```bash
# Dashboard completo
curl http://127.0.0.1:11534/debug/memory

# Solo valori chiave
curl -s http://127.0.0.1:11534/debug/memory | \
  jq '{allocata, riservata, libera}'

# ROCm tools
rocm-smi --showmeminfo vram
```

### Log Level

Per aumentare verbosità:
```bash
export OLLAMA_FAKE_LOGLEVEL=debug
```

---

## Estensioni Future

### Aggiungere Nuovo Modello

1. **Aggiungi in `index.html`:**
```javascript
const models = [
  // ... esistenti
  {
    label: 'Llama 3.1 8B',
    value: '/mnt/raid0/llama3.1-8b',
    legend: '<code>&lt;|begin_of_text|&gt;</code>...'
  }
]
```

2. **Aggiungi formatter in `prompt_formatter.py`:**
```python
def format_llama_prompt(messages: List[Dict]) -> str:
    # Implementa formatting specifico Llama
    ...

def detect_model_family(model_path: str) -> str:
    if "llama" in model_lower:
        return "llama"
    # ...
```

3. **Test:**
```bash
curl -X POST http://192.168.1.155:13010/api/chats \
  -H "Content-Type: application/json" \
  -d '{"model":"/mnt/raid0/llama3.1-8b"}'
```

### Export Conversazioni

In `main.py`:
```python
@app.get("/api/chats/{chat_id}/export")
async def export_chat(chat_id: str, format: str = "md"):
    chat = storage.get_chat(chat_id)
    if format == "md":
        return generate_markdown(chat)
    elif format == "json":
        return chat
```

---

**Fine Documentazione Tecnica**
