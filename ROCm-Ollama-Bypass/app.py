"""Ollama-compatible REST service using PyTorch on MI50."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from model_manager import ModelManager, parse_options
from rag_manager import RAGStore
from session_manager import ChatMessage, SessionManager
from utils import LOGGER, OllamaError, build_ndjson, configure_logging, detect_device, now_iso

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
configure_logging(level=os.environ.get("OLLAMA_FAKE_LOGLEVEL"), log_dir=os.environ.get("OLLAMA_FAKE_LOGDIR"))

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RAGQuery(BaseModel):
    dataset_id: str
    top_k: int = 3


class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    stream: bool = False
    options: Optional[Dict[str, Any]] = None
    rag: Optional[RAGQuery] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    stream: bool = False
    session_id: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    rag: Optional[RAGQuery] = None


class RAGDocument(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[Dict[str, Any]] = None


class RAGUpsertRequest(BaseModel):
    dataset_id: str
    documents: List[RAGDocument]


class RAGQueryRequest(BaseModel):
    dataset_id: str
    query: str
    top_k: int = 3


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    device_info = detect_device()
    LOGGER.info("Detected accelerator: %s (%s)", device_info["gpu_name"], device_info["device"])

    app = FastAPI(title="Ollama-Compatible PyTorch Service", version="0.1.0")

    app.state.device_info = device_info
    app.state.model_manager = ModelManager(
        device=device_info["device"],
        dtype=device_info["dtype"],
        quantization_allowed=False,
    )
    rag_dir = Path(__file__).parent / "rag_store"
    app.state.rag_store = RAGStore(rag_dir)
    app.state.session_manager = SessionManager()
    app.state.default_model = os.environ.get(
        "OLLAMA_FAKE_DEFAULT_MODEL", "Qwen/Qwen2.5-7B-Instruct"
    )

    register_routes(app)
    register_exception_handlers(app)
    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


def get_rag_store(request: Request) -> RAGStore:
    return request.app.state.rag_store


def ensure_model_loaded(manager: ModelManager, model_name: str, options: Optional[Dict[str, Any]]) -> None:
    quantize = None
    if options:
        quantize = options.get("quantize")
    manager.load_model(model_name, quantize=quantize)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def register_routes(app: FastAPI) -> None:
    @app.get("/api/version")
    async def get_version() -> Dict[str, Any]:
        device = app.state.device_info
        return {
            "version": app.version,
            "gpu": device["gpu_name"],
            "device": device["device"],
            "is_rocm": device["is_rocm"],
        }

    @app.get("/api/tags")
    async def list_tags(request: Request) -> Dict[str, Any]:
        manager = get_manager(request)
        return {"models": manager.list_models()}

    @app.get("/api/ps")
    async def list_processes(request: Request) -> Dict[str, Any]:
        manager = get_manager(request)
        return {"models": manager.list_models()}

    @app.get("/api/show")
    async def show_model(request: Request, name: str) -> Dict[str, Any]:
        manager = get_manager(request)
        ensure_model_loaded(manager, name, None)
        handle = manager.get_handle(name)
        return {
            "model": {
                "name": handle.name,
                "context_length": handle.context_length,
                "num_parameters": handle.num_parameters,
                "dtype": str(handle.dtype),
                "device": handle.device,
            }
        }

    @app.post("/api/delete")
    async def delete_model(request: Request) -> Dict[str, Any]:
        payload = await request.json()
        name = payload.get("model")
        if not name:
            raise HTTPException(status_code=400, detail="Missing model name")
        manager = get_manager(request)
        deleted = manager.unload_model(name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not loaded")
        return {"deleted": name}

    @app.post("/api/pull")
    async def pull_model() -> JSONResponse:
        return JSONResponse(status_code=501, content={"error": "Model registry not implemented"})

    @app.post("/api/generate")
    async def generate(request: Request, body: GenerateRequest):
        manager = get_manager(request)
        rag_store = get_rag_store(request)
        model_name = body.model or request.app.state.default_model
        ensure_model_loaded(manager, model_name, body.options)
        options = parse_options(body.options)

        prompt = body.prompt
        rag_metadata: Optional[Dict[str, Any]] = None
        if body.rag:
            rag_metadata = rag_store.build_augmented_prompt(
                dataset_id=body.rag.dataset_id,
                query=body.prompt,
                top_k=body.rag.top_k,
            )
            prompt = rag_metadata["augmented_prompt"]

        created_at = now_iso()

        if body.stream:

            def stream() -> Iterator[bytes]:
                accumulated = ""
                for fragment in manager.generate_stream(model_name, prompt, options):
                    if fragment:
                        accumulated += fragment
                        payload = {
                            "model": model_name,
                            "created_at": now_iso(),
                            "response": fragment,
                            "done": False,
                        }
                        yield build_ndjson(payload).encode("utf-8")
                    else:
                        break

                payload = {
                    "model": model_name,
                    "created_at": now_iso(),
                    "response": "",
                    "done": True,
                    "done_reason": "stop",
                    "context": {
                        "prompt": body.prompt,
                        "rag": rag_metadata,
                        "total_response": accumulated,
                    },
                }
                yield build_ndjson(payload).encode("utf-8")

            return StreamingResponse(stream(), media_type="application/x-ndjson")

        text = await asyncio.to_thread(manager.generate_sync, model_name, prompt, options)
        response = {
            "model": model_name,
            "created_at": created_at,
            "response": text,
            "done": True,
            "done_reason": "stop",
            "context": {
                "prompt": body.prompt,
                "rag": rag_metadata,
            },
        }
        return JSONResponse(response)

    @app.post("/api/chat")
    async def chat(request: Request, body: ChatRequest):
        manager = get_manager(request)
        sessions = get_session_manager(request)
        rag_store = get_rag_store(request)

        if not body.messages:
            raise HTTPException(status_code=400, detail="Chat requires at least one message")

        model_name = body.model or request.app.state.default_model
        ensure_model_loaded(manager, model_name, body.options)
        options = parse_options(body.options)

        session_id = body.session_id or str(uuid.uuid4())
        prompt = sessions.build_prompt(body.messages, body.session_id)

        rag_metadata: Optional[Dict[str, Any]] = None
        if body.rag:
            last_user_messages = [m for m in body.messages if m.normalised_role() == "user"]
            user_query = last_user_messages[-1].content if last_user_messages else ""
            rag_metadata = rag_store.build_augmented_prompt(
                dataset_id=body.rag.dataset_id,
                query=user_query,
                top_k=body.rag.top_k,
            )
            prompt = rag_metadata["augmented_prompt"] + "\n" + prompt

        async def streaming_chat() -> AsyncIterator[bytes]:
            accumulated = ""
            user_msg = next((m for m in reversed(body.messages) if m.normalised_role() == "user"), body.messages[-1])
            for fragment in manager.generate_stream(model_name, prompt, options):
                if fragment:
                    accumulated += fragment
                    payload = {
                        "model": model_name,
                        "created_at": now_iso(),
                        "response": fragment,
                        "done": False,
                        "session_id": session_id,
                    }
                    yield build_ndjson(payload).encode("utf-8")
                else:
                    break

            sessions.save_exchange(
                session_id,
                user_msg,
                ChatMessage(role="assistant", content=accumulated),
            )
            payload = {
                "model": model_name,
                "created_at": now_iso(),
                "response": "",
                "done": True,
                "done_reason": "stop",
                "session_id": session_id,
                "context": {"rag": rag_metadata},
            }
            yield build_ndjson(payload).encode("utf-8")

        if body.stream:
            return StreamingResponse(streaming_chat(), media_type="application/x-ndjson")

        text = await asyncio.to_thread(manager.generate_sync, model_name, prompt, options)
        user_msg = next((m for m in reversed(body.messages) if m.normalised_role() == "user"), body.messages[-1])
        sessions.save_exchange(
            session_id,
            user_msg,
            ChatMessage(role="assistant", content=text),
        )
        return {
            "model": model_name,
            "created_at": now_iso(),
            "message": {"role": "assistant", "content": text},
            "session_id": session_id,
            "done": True,
            "context": {"rag": rag_metadata},
        }

    @app.post("/rag/upsert")
    async def rag_upsert(request: Request, body: RAGUpsertRequest):
        count = get_rag_store(request).upsert(body.dataset_id, [doc.model_dump() for doc in body.documents])
        return {"dataset_id": body.dataset_id, "documents_indexed": count}

    @app.post("/rag/query")
    async def rag_query(request: Request, body: RAGQueryRequest):
        result = get_rag_store(request).build_augmented_prompt(
            dataset_id=body.dataset_id,
            query=body.query,
            top_k=body.top_k,
        )
        return result


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(OllamaError)
    async def handle_ollama_error(_: Request, exc: OllamaError):  # type: ignore[override]
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    @app.exception_handler(Exception)
    async def handle_generic(_: Request, exc: Exception):  # type: ignore[override]
        LOGGER.exception("Unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


app = create_app()

def main() -> None:
    parser = argparse.ArgumentParser(description="Ollama compatible PyTorch service")
    parser.add_argument("model", nargs="?", help="Model to preload on startup")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    os.environ.setdefault("OLLAMA_FAKE_DEFAULT_MODEL", args.model or app.state.default_model)
    if args.model:
        try:
            app.state.model_manager.load_model(args.model)
        except OllamaError as exc:
            LOGGER.error("Failed to preload model: %s", exc.detail)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
