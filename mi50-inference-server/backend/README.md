# MI50 Ollama-Compatible PyTorch Service

Questo progetto fornisce un micro-servizio REST compatibile con le API di Ollama, ma basato su **PyTorch + HuggingFace** e ottimizzato per GPU **AMD MI50 (ROCm)**. Consente di caricare modelli `transformers`, generare testo in modalità sincrona/streaming, mantenere sessioni chat stateful, ed esporre semplici endpoint RAG.

## Prerequisiti

- Python 3.10+
- ROCm/PyTorch con supporto MI50 (torch deve vedere `cuda`/`hip`).
- Connessione Internet (per scaricare i modelli la prima volta).

### Dipendenze Python

```bash
python3 -m venv ~/venvs/ollama_faidate
source ~/venvs/ollama_faidate/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota MI50:** non usiamo `bitsandbytes`. Se viene richiesto `quantize=8/4` l’API risponde con errore perché su ROCm non è supportato.

## Avvio del server

### Nota su /mnt/raid0

- I modelli già scaricati sul RAID sono in `/mnt/raid0` (es. `/mnt/raid0/qwen2.5-coder-7b-instruct`).
- Esporta le cache per evitare download/duplice spazio con:
  `export HF_HOME=/mnt/raid0/hf_cache`, `export TRANSFORMERS_CACHE=/mnt/raid0/hf_cache/transformers`, `export TORCH_HOME=/mnt/raid0/torch-cache`.
- Puoi avviare il server direttamente puntando al percorso locale: `python app.py "/mnt/raid0/qwen2.5-coder-7b-instruct"`.

```bash
source ~/venvs/ollama_faidate/bin/activate
python app.py "Qwen/Qwen2.5-7B-Instruct" --port 11534
```

> Puoi lasciare Ollama ufficiale in ascolto su 11434 (GPU M30) e usare questo servizio sulla MI50 su 11534. Se preferisci riutilizzare 11434, imposta `PORT=11434` quando il servizio originale è fermo.

### Avvio rapido (start.sh)

```bash
cd ~/mi50_come_ollama
./start.sh                   # verifica ambiente e avvia su 11534
# per cambiare porta: PORT=11535 ./start.sh
```

Argomenti disponibili:

```
usage: app.py [model] [--host HOST] [--port PORT] [--log-level LEVEL]
```

- `model`: (opzionale) modello da pre-caricare e usare come default.
- `--host`: indirizzo di binding (default `0.0.0.0`).
- `--port`: porta HTTP (default `11534`, per convivere con Ollama ufficiale su 11434).
- `--log-level`: livello log (`info`, `debug`, ...).

### Variabili utili

- `OLLAMA_FAKE_DEFAULT_MODEL`: modello fallback per le richieste che non specificano `model`.
- `OLLAMA_FAKE_LOGLEVEL`: livello di logging (`info` di default).
- `OLLAMA_FAKE_LOGDIR`: cartella log; di default puntiamo a `/dev/shm/mi50_ollama_logs` (ramdisk). Copia o sincronizza verso il RAID solo quando serve conservarli.
- `OLLAMA_FAKE_DEFAULT_MAX_NEW_TOKENS`, `OLLAMA_FAKE_DEFAULT_TEMPERATURE`, `OLLAMA_FAKE_DEFAULT_TOP_K`: preset inferenza "fast" (128 token greedy senza sampling).
- `OLLAMA_FAKE_STREAM_CHARS`: dimensione (in caratteri) dei chunk NDJSON emessi in streaming (`160`). Azzeralo per tornare al token-per-token.
- `OLLAMA_FAKE_MAX_PROMPT_TOKENS`: limite massimo per il prompt in ingresso (default 4096, viene tagliato lato server).
- `OLLAMA_FAKE_ATTN_IMPL`: implementazione dell'attenzione (`sdpa` per ROCm 5.x).
- `HIP_VISIBLE_DEVICES` / `PYTORCH_HIP_ALLOC_CONF`: variabili ROCm per fissare la GPU e ottimizzare l'allocator (`garbage_collection_threshold:0.85,max_split_size_mb:256`).

## Endpoint principali

| Endpoint                    | Metodo | Descrizione |
|----------------------------|--------|-------------|
| `/api/version`             | GET    | Informazioni su versione e GPU |
| `/api/tags` / `/api/ps`    | GET    | Lista dei modelli caricati |
| `/api/show?name=`          | GET    | Metadati di un modello |
| `/api/pull`                | POST   | Stub (501 Not Implemented) |
| `/api/delete`              | POST   | Unload del modello corrente |
| `/api/generate`            | POST   | Text generation (sync/streaming) |
| `/api/chat`                | POST   | Chat stateful con `session_id` |
| `/rag/upsert`              | POST   | Ingestione chunk RAG |
| `/rag/query`               | POST   | Recupero chunk e prompt aumentato |

### Esempi `curl`

**Versione**
```bash
curl http://localhost:11534/api/version
```

**Generazione sincrona**
```bash
curl -X POST http://localhost:11534/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Scrivi un haiku sulla pioggia"}'
```

**Streaming NDJSON**
```bash
curl -N -X POST http://localhost:11534/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Riassumi in 3 bullet il Rinascimento italiano","stream":true}'
```

**Chat con sessione**
```bash
curl -X POST http://localhost:11534/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","messages":[{"role":"user","content":"Ciao"}]}'
```

**Tool calling (OpenAI-style)**
```bash
curl -X POST http://localhost:11534/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
        "messages":[{"role":"user","content":"Esegui ls"}],
        "tools":[{
          "type":"function",
          "function":{
            "name":"execute_command",
            "description":"Esegue un comando shell",
            "parameters":{
              "type":"object",
              "properties":{"command":{"type":"string"}},
              "required":["command"]
            }
          }
        }],
        "stream":true
      }'
```

Lo stream NDJSON include eventi con `tool_calls` e `done_reason: "tool_calls"`. La risposta finale contiene `message.tool_calls` nel formato OpenAI; i client che parlano con il proxy locale possono propagare la chiamata di tool senza adattatori aggiuntivi.

**RAG**
```bash
# Upload documenti
curl -X POST http://localhost:11534/rag/upsert \
  -H 'Content-Type: application/json' \
  -d '{"dataset_id":"manuale","documents":[{"id":"par1","text":"Primo paragrafo"}]}'

# Query
curl -X POST http://localhost:11534/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"dataset_id":"manuale","query":"Che dice il paragrafo?"}'
```

## Test e validazione

### Test unitari/integrazione

```bash
./run_tests.sh
```

I test usano `fastapi.testclient` con stub per non caricare modelli reali.

### Smoke test manuale

Con il server attivo:

```bash
python smoke_test.py --host http://localhost:11534 --prompt "Dimmi una curiosità su Torino"
```

Aggiungi `--stream` per vedere l’output NDJSON.

## RAG Store

- Archivia i chunk in `rag_store/<dataset_id>.json`.
- Gli embedding vengono calcolati con `all-MiniLM-L6-v2` (CPU di default).
- Endpoint `generate`/`chat` accettano campo opzionale `"rag": {"dataset_id": ..., "top_k": ...}`: il prompt originale viene prefissato con il contesto recuperato.

## Note operative

- Il servizio forza `torch.float16` su ROCm/MI50; su CPU ricade a `float32`.
- I log JSON vivono in `/dev/shm/mi50_ollama_logs` (ramdisk): copia sul RAID solo quando devi conservarli (`cp /dev/shm/mi50_ollama_logs/mi50_ollama.log logs/`).
- Lo streaming NDJSON è raggruppato in chunk (~160 caratteri) per ridurre overhead Python/JSON; imposta `OLLAMA_FAKE_STREAM_CHARS=0` per tornare al token-per-token.
- Se richiedi `{"options":{"quantize":8}}` la risposta è un errore 400 con messaggio esplicito.
- `SessionManager` mantiene le chat in RAM (thread-safe). Per persistenza esterna è sufficiente sostituire l’implementazione con Redis/DB.
- Endpoint `/api/pull` e `/api/delete` sono presenti per compatibilità ma non scaricano modelli; `pull` restituisce 501.

## Deploy via systemd

Il repo include `systemd/mi50_ollama.service` già impostato per l'ambiente condiviso su `/mnt/raid0`.
Aggiorna il valore dell'`ExecStart` se vuoi un modello diverso oppure un indirizzo differente.

```bash
sudo cp systemd/mi50_ollama.service /etc/systemd/system/mi50_ollama.service
sudo systemctl daemon-reload
sudo systemctl enable --now mi50_ollama.service
```

Il service esporta automaticamente:
- cache HuggingFace/Torch su `/mnt/raid0/hf_cache` e `/mnt/raid0/torch_cache`;
- log JSON su `/dev/shm/mi50_ollama_logs/mi50_ollama.log` (ramdisk, copia manuale se serve archiviarli);
- preset ROCm (HIP_VISIBLE_DEVICES, PYTORCH_HIP_ALLOC_CONF) e preset inferenza fast (FP16, 128 token greedy);
- `PYTHONPATH` verso la cartella del servizio.

Controllo log:

```bash
journalctl -u mi50_ollama.service -f
TAILFILE=/dev/shm/mi50_ollama_logs/mi50_ollama.log
[ -f "$TAILFILE" ] && tail -f "$TAILFILE"
```
Per persistere i log: `cp /dev/shm/mi50_ollama_logs/mi50_ollama.log logs/` (o rsync verso il RAID).

### Flush automatico dei log

Per non dimenticare il flush, trovi anche `systemd/mi50_ollama-flush-logs.service` + `.timer` che eseguono `/scripts/flush_logs.sh` ogni 5 minuti.
Abilitazione (come root):

```bash
sudo cp systemd/mi50_ollama-flush-logs.service /etc/systemd/system/
sudo cp systemd/mi50_ollama-flush-logs.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mi50_ollama-flush-logs.timer
```

Puoi forzare un flush manuale con `./scripts/flush_logs.sh`.

## TODO / idee future

- Persistenza su disco della sessione chat.
- Auto-scaling su più GPU (MI50 + M40) con routing.
- Supporto a modelli quantizzati su M40 (bitsandbytes) mantenendo disaccoppiati i backend.

Buon divertimento con la MI50! 💚
### Client "chat_mi50"
### Server M40 (Ollama ufficiale)

Per avviare velocemente l'istanza Ollama sulla Tesla M40 usa la repo sorella `~/m40_ollama`:
```bash
cd ~/m40_ollama
PORT=11444 ./start.sh      # propone una porta libera se 11434 è occupata
```
I modelli sono salvati in `/mnt/raid0/ollama_models`. Ad esempio per scaricare `phi3:mini`:
```bash
cd ~/m40_ollama
export OLLAMA_HOST=127.0.0.1:11444
export OLLAMA_MODELS=/mnt/raid0/ollama_models
ollama pull phi3:mini
```
Gli orchestratori esterni possono collegarsi a `http://<server>:11444` (API Ollama standard).


Per una chat iterativa con editor locale puoi usare `chat_mi50.py`:

```bash
cd ~/mi50_come_ollama
./chat_mi50.py --host http://127.0.0.1:11534 --model /mnt/raid0/qwen2.5-coder-7b-instruct
```

Ogni turno apre `nano` (override con `CHAT_MI50_EDITOR`). Lo script applica il template Qwen (<|im_start|> ...), invia il prompt al servizio e salva il contesto in `chat_mi50_history.json`. Lasciare il buffer vuoto termina la sessione.

## Troubleshooting

Per problemi comuni e loro soluzioni, consulta [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).

Problemi risolti:
- **VRAM al 90% senza inferenza** - PyTorch pre-allocazione eccessiva (2025-10-18)

Per diagnostica VRAM in tempo reale:
```bash
curl http://localhost:11534/debug/memory | python3 -m json.tool
```

## Performance Optimization

### RAM-VRAM Balancing

Questo server ha **168GB RAM** che viene utilizzata strategicamente per ottimizzare VRAM e velocità inferenza.

**Strategia implementata:**
- Modello caricato in RAM (15GB su 168GB = 9% utilizzo)
- Transfer diretto RAM → VRAM senza buffer intermedi
- Cleanup automatico con `torch.cuda.empty_cache()`

**Risultati:**
- VRAM: 50% idle (16GB liberi per KV cache)
- Inferenza: 50-100 tokens/sec
- Latenza: 2-5s per risposte brevi

Per dettagli completi, consultare [RAM_VRAM_OPTIMIZATION.md](./RAM_VRAM_OPTIMIZATION.md).

**Nota importante:** Questa ottimizzazione è specifica per server con RAM abbondante (>64GB). Per server con meno RAM, vedere le strategie alternative nella documentazione.

### Quick Setup per Performance Ottimali

**Dopo installazione/riavvio server:**

```bash
# 1. Impostare GPU performance high (richiede sudo, una sola volta dopo boot)
sudo rocm-smi --setperflevel high

# 2. Verificare configurazione
rocm-smi
# Atteso: VRAM% ~45-50% (idle), Perf: high

# 3. Avviare servizio
cd ~/mi50_come_ollama
./start.sh

# 4. Test velocità (opzionale)
curl -H "Content-Type: application/json" http://localhost:11534/api/generate \
  -d "{\"model\": \"/mnt/raid0/qwen2.5-coder-7b-instruct\", \"prompt\": \"Say hello\", \"stream\": false}"
```

**Performance attese:**
- VRAM idle: 45-50%
- VRAM durante inferenza: 60-70%
- Velocità: 8-9 tokens/sec (120 token in ~15s)
- Temperatura GPU: 25-30°C idle, 40-50°C durante inferenza


## Compatibilità Ollama

Il server implementa le **API standard di Ollama** ed è compatibile con orchestratori come:
- ✅ **Goose** (testato, funzionante)
- ✅ LangChain / LlamaIndex
- ✅ Open WebUI
- ✅ Continue.dev / Cursor / Aider

**Endpoint:** `http://192.168.1.155:11534`

Per dettagli completi, vedere [COMPATIBILITY.md](./COMPATIBILITY.md).

