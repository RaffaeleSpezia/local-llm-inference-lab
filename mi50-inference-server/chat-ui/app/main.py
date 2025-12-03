from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .storage import ChatStorage
from .prompt_formatter import format_prompt_for_model
from .token_counter import count_messages_tokens, get_context_status

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
DATA_PATH = BASE_DIR / "chat_state.json"

INFERENCE_URL = os.environ.get("MI50_SERVER_URL", "http://127.0.0.1:11534").rstrip("/")
DEFAULT_MODEL = os.environ.get("CHAT_UI_DEFAULT_MODEL", "Qwen/Qwen2.5-7B-Instruct")

app = FastAPI(title="MI50 Chat UI", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

storage = ChatStorage(DATA_PATH, DEFAULT_MODEL)


class ChatCreateRequest(BaseModel):
    title: Optional[str] = None
    model: Optional[str] = None
    system_message: Optional[str] = None


class RenameRequest(BaseModel):
    title: str


class SystemMessageRequest(BaseModel):
    system_message: str


class GenerationOptions(BaseModel):
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    max_new_tokens: Optional[int] = 2048
    seed: Optional[int] = None


class SendMessageRequest(BaseModel):
    prompt: str
    options: Optional[GenerationOptions] = None
    raw_prompt: Optional[str] = None  # Se presente, usa questo invece di formattare automaticamente


class TrimRequest(BaseModel):
    strategy: str = "auto"  # "auto" | "sliding_window" | "to_target"
    keep_last_n: Optional[int] = None  # Per sliding_window
    target_percentage: Optional[float] = 0.5  # Per to_target


@app.on_event("startup")
async def startup() -> None:
    app.state.http = httpx.AsyncClient(timeout=None)


@app.on_event("shutdown")
async def shutdown() -> None:
    client: httpx.AsyncClient = app.state.http
    await client.aclose()


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/chats")
async def list_chats() -> Dict[str, Any]:
    return {"chats": storage.list_chats()}


@app.post("/api/chats")
async def create_chat(payload: ChatCreateRequest) -> Dict[str, Any]:
    chat = storage.create_chat(
        title=payload.title,
        model=payload.model,
        system_message=payload.system_message
    )
    return {"chat": chat}


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str) -> Dict[str, Any]:
    chat = storage.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat non trovata")
    return {"chat": chat}


@app.post("/api/chats/{chat_id}/rename")
async def rename_chat(chat_id: str, payload: RenameRequest) -> Dict[str, Any]:
    try:
        storage.update_title(chat_id, payload.title)
    except KeyError:
        raise HTTPException(status_code=404, detail="Chat non trovata") from None
    return {"status": "ok"}


@app.post("/api/chats/{chat_id}/system")
async def update_system_message(chat_id: str, payload: SystemMessageRequest) -> Dict[str, Any]:
    try:
        storage.update_system_message(chat_id, payload.system_message)
    except KeyError:
        raise HTTPException(status_code=404, detail="Chat non trovata") from None
    return {"status": "ok"}


# ========== NUOVI ENDPOINT PER TOKEN MANAGEMENT ==========

@app.get("/api/chats/{chat_id}/tokens")
async def get_token_stats(chat_id: str) -> Dict[str, Any]:
    """Ritorna statistiche sui token della conversazione."""
    try:
        stats = storage.get_token_stats(chat_id)
        return stats
    except KeyError:
        raise HTTPException(status_code=404, detail="Chat non trovata") from None


@app.post("/api/chats/{chat_id}/trim")
async def trim_chat(chat_id: str, payload: TrimRequest) -> Dict[str, Any]:
    """Taglia i messaggi della chat per ridurre l'uso del context."""
    try:
        if payload.strategy == "sliding_window":
            if not payload.keep_last_n:
                raise HTTPException(status_code=400, detail="keep_last_n richiesto per sliding_window")
            removed = storage.trim_messages_sliding_window(chat_id, payload.keep_last_n)
            return {
                "status": "ok",
                "strategy": "sliding_window",
                "removed_count": removed,
            }
        elif payload.strategy == "to_target":
            target_pct = payload.target_percentage or 0.5
            result = storage.trim_messages_to_target(chat_id, target_pct)
            return {
                "status": "ok",
                "strategy": "to_target",
                **result,
            }
        elif payload.strategy == "auto":
            # Auto: usa to_target con 50%
            result = storage.trim_messages_to_target(chat_id, 0.5)
            return {
                "status": "ok",
                "strategy": "auto",
                **result,
            }
        else:
            raise HTTPException(status_code=400, detail=f"Strategia sconosciuta: {payload.strategy}")
    except KeyError:
        raise HTTPException(status_code=404, detail="Chat non trovata") from None


@app.get("/api/chats/{chat_id}/trim/preview")
async def trim_preview(chat_id: str, keep_last_n: int = 10) -> Dict[str, Any]:
    """Anteprima dei messaggi dopo trim, senza applicare modifiche."""
    try:
        preview = storage.get_trimmed_messages_preview(chat_id, keep_last_n)
        return {
            "preview": preview,
            "message_count": len(preview),
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Chat non trovata") from None


# ========== SEND MESSAGE CON CONTROLLO TOKEN ==========

@app.post("/api/chats/{chat_id}/send")
async def send_message(chat_id: str, payload: SendMessageRequest, request: Request) -> StreamingResponse:
    try:
        chat = storage.ensure_chat(chat_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Chat non trovata") from None

    # Aggiungi messaggio utente
    storage.append_message(chat_id, "user", payload.prompt)

    # ========== CONTROLLO TOKEN ==========
    # Controlla token PRIMA di inviare al backend
    try:
        token_stats = storage.get_token_stats(chat_id)
        context_status = token_stats["context"]

        # Se CRITICAL (>95%), auto-trim automaticamente
        if context_status["status"] == "critical":
            # Auto-trim al 50% del context
            trim_result = storage.trim_messages_to_target(chat_id, target_percentage=0.5)
            context_warning = (
                f"⚠️ CONTEXT TRIMMED: Rimossi {trim_result['removed_count']} messaggi vecchi "
                f"({trim_result['tokens_before']} → {trim_result['tokens_after']} token). "
            )
        elif context_status["status"] == "warning":
            # WARNING (>80%): invia warning ma non trimmare
            context_warning = (
                f"⚠️ Context al {context_status['percentage']}% ({context_status['used']}/{context_status['limit']} token). "
                f"Considera di rimuovere messaggi vecchi. "
            )
        else:
            context_warning = None

    except Exception as e:
        # Se il conteggio token fallisce, continua comunque
        context_warning = None
        print(f"Warning: Token counting failed: {e}")

    # Costruisce il prompt formattato con i tag specifici del modello
    messages = storage.to_openai_messages(chat_id)

    # Se raw_prompt è fornito, usa quello (utente ha editato manualmente)
    if payload.raw_prompt:
        formatted_prompt = payload.raw_prompt
    else:
        formatted_prompt = format_prompt_for_model(chat["model"], messages)

    # Prepara le options per il backend
    options_dict = {}
    if payload.options:
        if payload.options.temperature is not None:
            options_dict["temperature"] = payload.options.temperature
        if payload.options.top_p is not None:
            options_dict["top_p"] = payload.options.top_p
        if payload.options.top_k is not None:
            options_dict["top_k"] = payload.options.top_k
        if payload.options.max_new_tokens is not None:
            options_dict["max_new_tokens"] = payload.options.max_new_tokens
        if payload.options.seed is not None and payload.options.seed > 0:
            options_dict["seed"] = payload.options.seed

    # Usa /api/generate con parametri
    request_payload = {
        "model": chat["model"],
        "prompt": formatted_prompt,
        "stream": True,
    }

    if options_dict:
        request_payload["options"] = options_dict

    client: httpx.AsyncClient = request.app.state.http

    async def stream() -> Any:
        # Se c'è un warning sul context, invialo come primo chunk
        if context_warning:
            yield json.dumps({
                "response": context_warning,
                "done": False,
                "context_warning": True
            }, ensure_ascii=False) + "\n"

        assistant_chunks: List[str] = []
        error_text: Optional[str] = None
        try:
            async with client.stream("POST", f"{INFERENCE_URL}/api/generate", json=request_payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # /api/generate usa "response" per i chunk
                    delta = data.get("response", "")
                    if delta:
                        assistant_chunks.append(delta)
                        # Converti in formato compatibile con il frontend
                        yield json.dumps({"response": delta, "done": data.get("done", False)}, ensure_ascii=False) + "\n"

                    # Quando done=true, invia il messaggio finale
                    if data.get("done"):
                        yield json.dumps({"done": True}, ensure_ascii=False) + "\n"

        except httpx.HTTPError as exc:
            error_text = f"Errore backend: {exc}"
            yield json.dumps({"error": error_text}, ensure_ascii=False) + "\n"
        finally:
            text = "".join(assistant_chunks)
            if not text and error_text:
                text = error_text
            storage.append_message(chat_id, "assistant", text)

    return StreamingResponse(stream(), media_type="application/x-ndjson")
