# Chat UI Setup - Web Interface

**Versione:** 1.0
**Data:** 3 Dicembre 2025
**Autore:** Raffaele Spezia

---

## Indice

1. [Overview](#overview)
2. [Architettura](#architettura)
3. [Installazione](#installazione)
4. [Configurazione](#configurazione)
5. [Features](#features)
6. [API Endpoints](#api-endpoints)
7. [Avvio](#avvio)
8. [Customizzazione](#customizzazione)

---

## Overview

### Che Cos'è

**Chat UI** è un'interfaccia web moderna per interagire con i modelli LLM del backend MI50.

**Features principali:**
- Chat stateful multi-sessione
- Selezione modello dinamica
- Streaming response real-time
- Parameter tuning (temperature, tokens, etc.)
- Token counting e context management
- Trimming automatico conversazioni
- Export/import chat JSON
- System message configurabile

### Stack Tecnologico

| Layer | Tecnologia | Ruolo |
|-------|-----------|-------|
| **Backend** | FastAPI 0.122 | REST API |
| **Frontend** | Vanilla JS + HTML5 | UI single-page |
| **HTTP Client** | httpx (async) | Proxy a Backend MI50 |
| **Storage** | JSON file locale | Salvataggio chat |
| **Styling** | CSS3 custom | UI minimale |

---

## Architettura

### Component Diagram

```
┌───────────────────────────────────────────────────┐
│           Browser (User)                          │
│                                                   │
│  ┌────────────────────────────────────────────┐  │
│  │  static/index.html (Frontend)              │  │
│  │  - Chat UI                                 │  │
│  │  - Model selector                          │  │
│  │  - Token counter                           │  │
│  │  - Parameter sliders                       │  │
│  └────────────┬───────────────────────────────┘  │
└───────────────┼──────────────────────────────────┘
                │ HTTP/SSE
                │
┌───────────────▼──────────────────────────────────┐
│  Chat UI Service (Port 12000)                    │
│                                                   │
│  ┌────────────────────────────────────────────┐  │
│  │  app/main.py (FastAPI)                     │  │
│  │  ┌──────────────────────────────────────┐  │  │
│  │  │ Endpoints:                           │  │  │
│  │  │  GET  /                              │  │  │
│  │  │  GET  /api/chats                     │  │  │
│  │  │  POST /api/chats                     │  │  │
│  │  │  POST /api/chats/{id}/send           │  │  │
│  │  │  GET  /api/chats/{id}/tokens         │  │  │
│  │  │  POST /api/chats/{id}/trim           │  │  │
│  │  └──────────────────────────────────────┘  │  │
│  │                                             │  │
│  │  app/storage.py (ChatStorage)              │  │
│  │  - Persist chat to JSON                    │  │
│  │  - Token counting                          │  │
│  │  - Trim strategies                         │  │
│  │                                             │  │
│  │  app/prompt_formatter.py                   │  │
│  │  - Format per Qwen/Gemma/DeepSeek          │  │
│  └────────────┬────────────────────────────────┘  │
└───────────────┼──────────────────────────────────┘
                │ HTTP Proxy
                │
┌───────────────▼──────────────────────────────────┐
│  Backend MI50 (Port 11534)                       │
│  /api/generate, /api/chat                        │
└──────────────────────────────────────────────────┘
```

### Data Flow

**Messaggio Utente:**
```
1. User → Type message in browser
2. Browser → POST /api/chats/{id}/send {"prompt": "..."}
3. Chat UI → format_prompt_for_model(messages, model)
4. Chat UI → POST {INFERENCE_URL}/api/generate {"prompt": formatted}
5. Backend MI50 → Stream NDJSON response
6. Chat UI → Proxy stream a browser
7. Browser → Display streaming text
8. Chat UI → Save message + response to chat_state.json
```

---

## Installazione

### Prerequisiti

**Server MI50 Backend deve essere running:**
```bash
curl http://127.0.0.1:11534/api/version
# Se fallisce → avvia backend prima
```

### Setup Venv

```bash
ssh lele2@192.168.1.155
cd ~/mi50_stack

# Crea directory
mkdir -p mi50_chat_ui
cd mi50_chat_ui

# Usa venv condiviso
source /mnt/raid0/shared_envs/venv-rocm311/bin/activate

# Install dependencies
pip install fastapi uvicorn httpx aiofiles jinja2
```

### File Structure

```
mi50_chat_ui/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app (350 righe)
│   ├── storage.py              # ChatStorage class (400 righe)
│   ├── prompt_formatter.py     # Format prompts per modello (150 righe)
│   └── token_counter.py        # Token counting (200 righe)
│
├── static/
│   ├── index.html              # Single-page UI (800 righe)
│   └── style.css               # (opzionale, inline in HTML)
│
├── start_chat_ui.sh            # Avvio script
├── requirements.txt
└── chat_state.json             # Generated: chat storage
```

### Copia File

**Da backup recuperato:**

```bash
cd ~/mi50_stack
cp -r /tmp/mi50_chat_ui_backup/* ~/mi50_stack/mi50_chat_ui/

# Verifica
ls -la ~/mi50_stack/mi50_chat_ui/app/
ls -la ~/mi50_stack/mi50_chat_ui/static/
```

### Requirements.txt

```txt
fastapi==0.122.0
uvicorn[standard]==0.32.0
httpx==0.28.0
aiofiles==24.1.0
jinja2==3.1.4
pydantic==2.10.1
```

Installa:
```bash
cd ~/mi50_stack/mi50_chat_ui
pip install -r requirements.txt
```

---

## Configurazione

### Variabili Ambiente

**File:** `start_chat_ui.sh` o `~/.bashrc`

```bash
# Backend MI50 URL
export MI50_SERVER_URL="http://127.0.0.1:11534"

# Default model (usato per nuove chat)
export CHAT_UI_DEFAULT_MODEL="/mnt/raid0/qwen2.5-coder-7b-instruct"

# Port (default 12000)
export CHAT_UI_PORT="12000"

# Host binding
export CHAT_UI_HOST="0.0.0.0"  # Allow remote connections
```

### Script start_chat_ui.sh

```bash
#!/bin/bash

# Activate venv
source /mnt/raid0/shared_envs/venv-rocm311/bin/activate

# Export config
export MI50_SERVER_URL="http://127.0.0.1:11534"
export CHAT_UI_DEFAULT_MODEL="/mnt/raid0/qwen2.5-coder-7b-instruct"

# Port selection
PORT=${CHAT_UI_PORT:-12000}

# Check if port busy
if lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
    echo "⚠️  Port $PORT already in use"
    exit 1
fi

echo "Starting Chat UI on port $PORT..."
echo "Backend: $MI50_SERVER_URL"

# Start uvicorn
cd "$(dirname "$0")"
uvicorn app.main:app --host 0.0.0.0 --port $PORT --log-level info
```

Rendi eseguibile:
```bash
chmod +x ~/mi50_stack/mi50_chat_ui/start_chat_ui.sh
```

---

## Features

### 1. Multi-Chat Sessions

**Gestione sessioni:**
- Crea nuova chat (pulsante "+ New Chat")
- Lista chat sidebar (ordinata per data)
- Switch tra chat (mantiene context separato)
- Rename chat (double-click su titolo)
- Delete chat (pulsante X)

**Storage:**
```json
// chat_state.json
{
  "chats": {
    "chat-uuid-1": {
      "id": "chat-uuid-1",
      "title": "Python Help",
      "model": "/mnt/raid0/qwen2.5-coder-7b-instruct",
      "system_message": "You are a helpful Python assistant",
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "How do list comprehensions work?"},
        {"role": "assistant", "content": "List comprehensions provide..."}
      ],
      "created_at": "2025-12-03T10:30:00Z",
      "updated_at": "2025-12-03T10:35:00Z"
    }
  },
  "current_chat_id": "chat-uuid-1"
}
```

### 2. Model Selection

**Dropdown models configurati:**
```javascript
const MODELS = [
  {
    value: "/mnt/raid0/qwen2.5-coder-7b-instruct",
    label: "Qwen2.5 Coder 7B ⭐",
    contextLength: 32768
  },
  {
    value: "/mnt/raid0/qwen2.5-coder-14b-instruct",
    label: "Qwen2.5 Coder 14B",
    contextLength: 32768
  },
  {
    value: "/mnt/raid0/gemma-2-9b-it",
    label: "Gemma 2 9B IT",
    contextLength: 8192
  },
  {
    value: "/mnt/raid0/deepseek-coder-6.7b-instruct",
    label: "DeepSeek Coder 6.7B",
    contextLength: 16384
  },
  {
    value: "/mnt/raid0/gemma3-4b-it",
    label: "Gemma3 4B IT",
    contextLength: 8192
  }
];
```

**Cambio modello:**
- Dropdown in header
- Switch tra chat mantiene modello specifico
- Nuova chat usa modello selezionato

### 3. Streaming Response

**Real-time text generation:**

```javascript
// In static/index.html
async function sendMessage(chatId, prompt, options) {
  const response = await fetch(`/api/chats/${chatId}/send`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt, options})
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const {done, value} = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (!line.trim()) continue;
      const data = JSON.parse(line);

      if (data.response) {
        appendToMessage(data.response);  // Update UI
      }
    }
  }
}
```

**UI durante streaming:**
- Animated "..." indicator
- Progressive text append
- Auto-scroll to bottom
- Stop button (abort request)

### 4. Parameter Tuning

**Sliders in UI:**

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Temperature** | 0.0 - 2.0 | 0.7 | Randomness (0=greedy) |
| **Top-P** | 0.0 - 1.0 | 0.9 | Nucleus sampling |
| **Top-K** | 0 - 100 | 50 | Top-K sampling |
| **Max Tokens** | 128 - 4096 | 2048 | Max output length |
| **Seed** | null/int | null | Reproducibility |

**Salvataggio:**
- Parameter salvati per chat (not global)
- Reset a default (pulsante "Reset")

### 5. Token Counting

**Display real-time:**
```
Context: 1250 / 32768 tokens (3.8%)
[████░░░░░░░░░░░░░░░░░░░░░░░]
```

**Conteggio:**
- System message tokens
- User messages tokens
- Assistant responses tokens
- Total context used
- Percentuale vs context limit

**Warning thresholds:**
- 70-90%: ⚠️ Giallo (vicino al limite)
- 90%+: 🔴 Rosso (trim necessario)

### 6. Context Trimming

**3 strategie:**

**a) Auto (default):**
- Se context > 90% → auto-trim a 50%
- Mantiene system message + ultimi N messaggi

**b) Sliding Window:**
```json
{
  "strategy": "sliding_window",
  "keep_last_n": 10  // Keep last 10 messages (5 turni)
}
```
- Rimuove messaggi più vecchi
- Mantiene ultimi N

**c) To Target:**
```json
{
  "strategy": "to_target",
  "target_percentage": 0.5  // Trim to 50% of context
}
```
- Riduce a percentuale target
- Algoritmo: rimuove messaggi dal più vecchio finché < target

**UI:**
- Pulsante "Trim" in header
- Dialog modal per selezione strategia
- Conferma con preview tokens risparmiati

### 7. System Message

**Configurazione:**
- Input text area in settings panel
- Editable per chat corrente
- Default: "You are a helpful assistant"
- Salvato in chat_state.json

**Prompt format esempi:**
```
System: You are a Python expert focused on clean, idiomatic code.
User: Write a binary search
Assistant: ...
```

### 8. Export/Import

**Export chat:**
```javascript
// Download JSON
function exportChat(chatId) {
  const chat = getChat(chatId);
  const blob = new Blob([JSON.stringify(chat, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `chat-${chat.title}-${Date.now()}.json`;
  a.click();
}
```

**Import chat:**
- Upload JSON file
- Validate schema
- Add to chat list

---

## API Endpoints

### Base URL

```
http://192.168.1.155:12000
```

### GET /

**Descrizione:** Serve UI HTML

**Response:** `static/index.html`

**Esempio:**
```bash
curl http://192.168.1.155:12000
# O in browser: http://192.168.1.155:12000
```

### GET /api/chats

**Descrizione:** Lista tutte le chat

**Response:**
```json
{
  "chats": [
    {
      "id": "chat-uuid-1",
      "title": "Python Help",
      "model": "/mnt/raid0/qwen2.5-coder-7b-instruct",
      "created_at": "2025-12-03T10:00:00Z",
      "updated_at": "2025-12-03T10:30:00Z",
      "message_count": 6
    }
  ]
}
```

### POST /api/chats

**Descrizione:** Crea nuova chat

**Request:**
```json
{
  "title": "New Chat",
  "model": "/mnt/raid0/qwen2.5-coder-7b-instruct",
  "system_message": "You are a helpful assistant"
}
```

**Response:**
```json
{
  "chat": {
    "id": "chat-uuid-new",
    "title": "New Chat",
    "model": "/mnt/raid0/qwen2.5-coder-7b-instruct",
    "system_message": "...",
    "messages": [],
    "created_at": "2025-12-03T11:00:00Z"
  }
}
```

### POST /api/chats/{id}/send

**Descrizione:** Invia messaggio + ottieni risposta streaming

**Request:**
```json
{
  "prompt": "Explain list comprehensions",
  "options": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_new_tokens": 2048
  }
}
```

**Response:** Streaming NDJSON
```json
{"response": "List ", "done": false}
{"response": "comprehensions ", "done": false}
{"response": "are a concise way...", "done": false}
{"response": "", "done": true, "total_tokens": 245}
```

### GET /api/chats/{id}/tokens

**Descrizione:** Token stats per chat

**Response:**
```json
{
  "total_tokens": 1250,
  "context_length": 32768,
  "percentage": 3.8,
  "status": "ok",
  "breakdown": {
    "system_tokens": 15,
    "user_tokens": 420,
    "assistant_tokens": 815
  }
}
```

### POST /api/chats/{id}/trim

**Descrizione:** Trim messaggi chat

**Request:**
```json
{
  "strategy": "sliding_window",
  "keep_last_n": 10
}
```

**Response:**
```json
{
  "status": "ok",
  "strategy": "sliding_window",
  "removed_count": 8,
  "tokens_before": 2800,
  "tokens_after": 1150
}
```

---

## Avvio

### Manuale (Foreground)

```bash
ssh lele2@192.168.1.155
cd ~/mi50_stack/mi50_chat_ui

# Activate venv
source /mnt/raid0/shared_envs/venv-rocm311/bin/activate

# Set backend URL
export MI50_SERVER_URL="http://127.0.0.1:11534"

# Start
uvicorn app.main:app --host 0.0.0.0 --port 12000 --log-level info
```

### Via Script (Foreground)

```bash
cd ~/mi50_stack/mi50_chat_ui
./start_chat_ui.sh
```

### Background (nohup)

```bash
cd ~/mi50_stack/mi50_chat_ui
nohup ./start_chat_ui.sh > /tmp/mi50_services_logs/chat_ui.log 2>&1 &

# Check log
tail -f /tmp/mi50_services_logs/chat_ui.log
```

### Via start_all_services.sh

```bash
cd ~/mi50_stack
./start_all_services.sh
# → Avvia backend + chat UI + dashboard in background
```

### Accesso UI

**Da browser:**
```
http://192.168.1.155:12000
```

**Features disponibili:**
- Chat interface
- Model selector
- Token counter
- Parameter tuning
- Chat management

---

## Customizzazione

### Aggiungere Modello

**1. Aggiorna models array in `static/index.html`:**

```javascript
const MODELS = [
  // ... existing models
  {
    value: "/mnt/raid0/nuovo-modello",
    label: "Nuovo Modello 10B",
    contextLength: 16384
  }
];
```

**2. Aggiorna prompt formatter in `app/prompt_formatter.py`:**

```python
def format_prompt_for_model(messages, model_name):
    # ... existing formats

    if "nuovo-modello" in model_name:
        return format_custom_prompt(messages)

    # Fallback
    return format_chatml(messages)
```

### Cambiare System Message Default

**Edit `app/storage.py`:**

```python
DEFAULT_SYSTEM_MESSAGE = "You are a specialized coding assistant focused on Python and JavaScript."
```

### Cambiare Theme UI

**Edit CSS in `static/index.html`:**

```css
:root {
  --bg-primary: #1e1e1e;        /* Dark background */
  --bg-secondary: #2d2d2d;      /* Sidebar */
  --text-primary: #e0e0e0;      /* Text */
  --accent: #4a9eff;            /* Accent color */
}
```

### Aggiungere Custom Endpoint

**In `app/main.py`:**

```python
@app.post("/api/chats/{chat_id}/summarize")
async def summarize_chat(chat_id: str) -> Dict[str, Any]:
    chat = storage.get_chat(chat_id)

    # Call backend to generate summary
    summary_prompt = f"Summarize this conversation:\n{chat['messages']}"
    response = await app.state.http.post(
        f"{INFERENCE_URL}/api/generate",
        json={"prompt": summary_prompt, "max_new_tokens": 200}
    )

    return {"summary": response.json()["response"]}
```

---

## Next Steps

Con Chat UI configurata, procedi a:

**→ [05-dashboard-setup.md](./05-dashboard-setup.md)** - Setup dashboard monitoring

---

**Licenza:** CC BY-NC-SA 4.0
**Autore:** Raffaele Spezia
**Repository:** https://github.com/RaffaeleSpezia/local-llm-inference-lab
