"""Ollama-compatible REST service using PyTorch on MI50."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from model_manager import ModelManager, parse_options, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from rag_manager import RAGStore
from session_manager import ChatMessage, SessionManager
from tool_manager import IncompleteToolCallError, InvalidToolCallError, ToolManager
from utils import LOGGER, OllamaError, build_ndjson, configure_logging, detect_device, now_iso
from token_broadcaster import TokenBroadcaster

import torch
import json


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _token_preview(delta: str, limit: int) -> str:
    preview = delta.replace("\r", "\\r").replace("\n", "\\n")
    if limit > 0 and len(preview) > limit:
        if limit > 3:
            preview = preview[: limit - 3] + "..."
        else:
            preview = preview[:limit]
    return preview


async def _token_log_worker(app: FastAPI) -> None:
    broadcaster: TokenBroadcaster = app.state.token_broadcaster
    queue = broadcaster.subscribe()
    limit = getattr(app.state, "token_log_char_limit", 160)
    try:
        while True:
            event = await queue.get()
            if event.get("type") != "token":
                continue
            delta = event.get("delta", "")
            if not delta:
                continue
            endpoint = event.get("endpoint", "?")
            req_id = event.get("request_id", "-")
            snippet = _token_preview(delta, limit)
            LOGGER.info(
                "[stream][%s][%s] %s",
                endpoint,
                req_id[:8],
                snippet,
            )
    except asyncio.CancelledError:
        raise
    finally:
        broadcaster.unsubscribe(queue)


def get_vram_usage():
    """Get current VRAM usage in GB"""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return {"allocated_gb": round(allocated, 2), "reserved_gb": round(reserved, 2)}
    except:
        pass
    return {"allocated_gb": 0, "reserved_gb": 0}


def log_request_details(endpoint, prompt_or_messages, tools=None, options=None):
    """Log detailed info about incoming request"""
    if isinstance(prompt_or_messages, str):
        prompt_len = len(prompt_or_messages)
        LOGGER.info(f"[{endpoint}] Prompt length: {prompt_len} chars (~{prompt_len//4} tokens)")
        LOGGER.info(f"[{endpoint}] Prompt preview (first 500 chars): {prompt_or_messages[:500]}")
    else:
        total_len = sum(len(str(m.content)) for m in prompt_or_messages)
        LOGGER.info(f"[{endpoint}] Messages: {len(prompt_or_messages)}, Total length: {total_len} chars (~{total_len//4} tokens)")
        for i, msg in enumerate(prompt_or_messages):
            content_preview = str(msg.content)[:200] if hasattr(msg, 'content') else str(msg)[:200]
            LOGGER.info(f"[{endpoint}] Message {i+1} [{msg.role if hasattr(msg, 'role') else 'unknown'}]: {content_preview}...")

    if tools:
        LOGGER.info(f"[{endpoint}] Tools defined: {len(tools)} (adds ~1500-2000 tokens to prompt)")
        for i, tool in enumerate(tools[:3]):  # Primi 3 tools
            tool_name = tool.get('function', {}).get('name', 'unknown') if isinstance(tool, dict) else 'unknown'
            LOGGER.info(f"[{endpoint}]   Tool {i+1}: {tool_name}")

    if options:
        LOGGER.info(f"[{endpoint}] Options: {options}")

    vram = get_vram_usage()
    LOGGER.info(f"[{endpoint}] VRAM before: {vram['allocated_gb']}GB allocated, {vram['reserved_gb']}GB reserved")


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
    tools: Optional[List[Dict[str, Any]]] = None


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
    app.state.token_broadcaster = TokenBroadcaster()
    app.state.log_tokens = _env_flag("OLLAMA_FAKE_LOG_TOKENS", True)
    app.state.token_log_char_limit = _env_int("OLLAMA_FAKE_LOG_TOKEN_CHARS", 160)
    app.state.token_log_task = None

    @app.on_event("startup")
    async def _setup_broadcaster() -> None:
        loop = asyncio.get_running_loop()
        app.state.token_broadcaster.set_loop(loop)
        if app.state.log_tokens:
            app.state.token_log_task = asyncio.create_task(_token_log_worker(app))
        else:
            app.state.token_log_task = None

    @app.on_event("shutdown")
    async def _shutdown_token_logger() -> None:
        task = getattr(app.state, "token_log_task", None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

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


def broadcast_event(app: FastAPI, payload: Dict[str, Any]) -> None:
    broadcaster = getattr(app.state, 'token_broadcaster', None)
    if broadcaster:
        broadcaster.publish(payload)


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


    @app.websocket("/ws/tokens")
    async def tokens_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        broadcaster: TokenBroadcaster = app.state.token_broadcaster
        queue = broadcaster.subscribe()
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event)
        except WebSocketDisconnect:
            pass
        finally:
            broadcaster.unsubscribe(queue)


    @app.get("/debug/memory")
    async def debug_memory() -> Dict[str, Any]:
        """Debug endpoint per analisi VRAM."""
        import torch
        info = {}
        
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            
            # Memoria allocata vs riservata
            allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
            max_allocated_gb = torch.cuda.max_memory_allocated(0) / (1024**3)
            
            info["memory"] = {
                "allocated_gb": round(allocated_gb, 2),
                "reserved_gb": round(reserved_gb, 2),
                "max_allocated_gb": round(max_allocated_gb, 2),
                "waste_gb": round(reserved_gb - allocated_gb, 2)
            }
            
            # Statistiche dettagliate
            stats = torch.cuda.memory_stats(0)
            info["stats"] = {
                "num_alloc_retries": stats.get("num_alloc_retries", 0),
                "num_ooms": stats.get("num_ooms", 0),
                "active_bytes": stats.get("active_bytes.all.current", 0) / (1024**3),
                "reserved_bytes": stats.get("reserved_bytes.all.current", 0) / (1024**3),
            }
        else:
            info["cuda_available"] = False
            
        return info
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

        log_request_details("generate", body.prompt, options=body.options)

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
        request_id = str(uuid.uuid4())
        broadcast_event(request.app, {
            "type": "start",
            "endpoint": "generate",
            "request_id": request_id,
            "model": model_name,
            "prompt": prompt,
            "original_prompt": body.prompt,
            "prompt_length": len(prompt),
            "options": body.options or {},
            "ts": created_at,
        })

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
                        broadcast_event(request.app, {
                            "type": "token",
                            "endpoint": "generate",
                            "request_id": request_id,
                            "model": model_name,
                            "delta": fragment,
                            "ts": now_iso(),
                        })
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
                broadcast_event(request.app, {
                    "type": "done",
                    "endpoint": "generate",
                    "request_id": request_id,
                    "model": model_name,
                    "ts": now_iso(),
                    "done_reason": "stop",
                    "total_response": accumulated,
                })
                yield build_ndjson(payload).encode("utf-8")

            return StreamingResponse(stream(), media_type="application/x-ndjson")

        text = await asyncio.to_thread(manager.generate_sync, model_name, prompt, options)

        # Log completion and cleanup memory
        LOGGER.info(f"[generate] Completed. Response length: {len(text)} chars")
        vram_after = get_vram_usage()
        LOGGER.info(f"[generate] VRAM after: {vram_after['allocated_gb']}GB allocated, {vram_after['reserved_gb']}GB reserved")
        torch.cuda.empty_cache()

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
        broadcast_event(request.app, {
            "type": "done",
            "endpoint": "generate",
            "request_id": request_id,
            "model": model_name,
            "ts": now_iso(),
            "done_reason": "stop",
            "total_response": text,
        })
        return JSONResponse(response)

    @app.post("/api/chat")
    async def chat(request: Request, body: ChatRequest):
        manager = get_manager(request)
        sessions = get_session_manager(request)
        rag_store = get_rag_store(request)
        tool_manager = ToolManager(body.tools)

        if not body.messages:
            raise HTTPException(status_code=400, detail="Chat requires at least one message")

        model_name = body.model or request.app.state.default_model
        ensure_model_loaded(manager, model_name, body.options)
        options = parse_options(body.options)

        log_request_details("chat", body.messages, tools=body.tools, options=body.options)

        session_id = body.session_id or str(uuid.uuid4())

        messages_for_prompt: List[ChatMessage] = list(body.messages)
        if tool_manager.has_tools():
            messages_for_prompt = [
                ChatMessage(role="system", content=tool_manager.build_system_prompt()),
                *messages_for_prompt,
            ]

        prompt = sessions.build_prompt(messages_for_prompt, body.session_id)

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

        user_msg = next((m for m in reversed(body.messages) if m.normalised_role() == "user"), body.messages[-1])
        chat_request_id = str(uuid.uuid4())
        broadcast_event(request.app, {
            "type": "start",
            "endpoint": "chat",
            "request_id": chat_request_id,
            "model": model_name,
            "session_id": session_id,
            "prompt": prompt,
            "prompt_length": len(prompt),
            "options": body.options or {},
            "ts": now_iso(),
        })

        async def streaming_chat() -> AsyncIterator[bytes]:
            accumulated = ""
            buffer = ""
            mode = "pending" if tool_manager.has_tools() else "disabled"
            tool_calls_openai: Optional[List[Dict[str, Any]]] = None
            tool_raw_message = ""

            def emit_text(fragment: str) -> bytes:
                broadcast_event(request.app, {
                    "type": "token",
                    "endpoint": "chat",
                    "request_id": chat_request_id,
                    "model": model_name,
                    "session_id": session_id,
                    "delta": fragment,
                    "ts": now_iso(),
                })
                payload = {
                    "model": model_name,
                    "created_at": now_iso(),
                    "response": fragment,
                    "done": False,
                    "session_id": session_id,
                }
                return build_ndjson(payload).encode("utf-8")

            for fragment in manager.generate_stream(model_name, prompt, options):
                if not fragment:
                    break

                if tool_manager.has_tools() and tool_calls_openai is None and mode != "disabled":
                    buffer += fragment
                    stripped = buffer.lstrip()

                    if mode == "pending":
                        if not stripped:
                            continue
                        if stripped[0] in "{[":
                            mode = "tool"
                        else:
                            mode = "disabled"
                            accumulated += buffer
                            yield emit_text(buffer)
                            buffer = ""
                            continue

                    if mode == "tool":
                        try:
                            calls = tool_manager.parse_tool_calls(stripped)
                        except IncompleteToolCallError:
                            continue
                        except InvalidToolCallError:
                            mode = "disabled"
                            accumulated += buffer
                            yield emit_text(buffer)
                            buffer = ""
                            continue
                        else:
                            tool_calls_openai = tool_manager.to_openai_calls(calls)
                            tool_raw_message = stripped
                            payload = {
                                "model": model_name,
                                "created_at": now_iso(),
                                "response": "",
                                "tool_calls": tool_calls_openai,
                                "done": False,
                                "session_id": session_id,
                            }
                            yield build_ndjson(payload).encode("utf-8")
                            buffer = ""
                            continue

                accumulated += fragment
                yield emit_text(fragment)

            if buffer and tool_calls_openai is None:
                accumulated += buffer
                yield emit_text(buffer)

            assistant_content = tool_raw_message if tool_calls_openai else accumulated
            sessions.save_exchange(
                session_id,
                user_msg,
                ChatMessage(role="assistant", content=assistant_content),
            )

            done_payload: Dict[str, Any] = {
                "model": model_name,
                "created_at": now_iso(),
                "response": "",
                "done": True,
                "done_reason": "tool_calls" if tool_calls_openai else "stop",
                "session_id": session_id,
                "context": {"rag": rag_metadata},
            }
            if tool_calls_openai:
                done_payload["message"] = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls_openai,
                }
                done_payload["tool_calls"] = tool_calls_openai

            broadcast_event(request.app, {
                "type": "done",
                "endpoint": "chat",
                "request_id": chat_request_id,
                "model": model_name,
                "session_id": session_id,
                "ts": now_iso(),
                "done_reason": done_payload["done_reason"],
                "total_response": assistant_content,
            })

            yield build_ndjson(done_payload).encode("utf-8")

        if body.stream:
            return StreamingResponse(streaming_chat(), media_type="application/x-ndjson")

        text = await asyncio.to_thread(manager.generate_sync, model_name, prompt, options)
        assistant_message = ChatMessage(role="assistant", content=text)

        if tool_manager.has_tools():
            try:
                calls = tool_manager.parse_tool_calls(text)
            except IncompleteToolCallError:
                calls = None
            except InvalidToolCallError:
                calls = None
            else:
                tool_calls_openai = tool_manager.to_openai_calls(calls)
                sessions.save_exchange(session_id, user_msg, assistant_message)
                broadcast_event(request.app, {
                    "type": "done",
                    "endpoint": "chat",
                    "request_id": chat_request_id,
                    "model": model_name,
                    "session_id": session_id,
                    "ts": now_iso(),
                    "done_reason": "tool_calls",
                })
                return {
                    "model": model_name,
                    "created_at": now_iso(),
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls_openai,
                    },
                    "session_id": session_id,
                    "done": True,
                    "done_reason": "tool_calls",
                    "context": {"rag": rag_metadata},
                }

        sessions.save_exchange(session_id, user_msg, assistant_message)
        broadcast_event(request.app, {
            "type": "done",
            "endpoint": "chat",
            "request_id": chat_request_id,
            "model": model_name,
            "session_id": session_id,
            "ts": now_iso(),
            "done_reason": "stop",
            "total_response": text,
        })
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



    # ============================================================
    # OpenAI-compatible endpoints (v1 API)
    # ============================================================
    
    @app.get("/v1/models")
    async def openai_list_models(request: Request) -> Dict[str, Any]:
        """OpenAI-compatible models list endpoint."""
        manager = get_manager(request)
        models = manager.list_models()
        
        # Convert to OpenAI format
        openai_models = []
        for model in models:
            openai_models.append({
                "id": model["name"],
                "object": "model",
                "created": 0,
                "owned_by": "mi50-server",
                "permission": [],
                "root": model["name"],
                "parent": None
            })
        
        return {
            "object": "list",
            "data": openai_models
        }
    
    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request: Request) -> Dict[str, Any]:
        """OpenAI-compatible chat completions endpoint."""
        payload = await request.json()
        
        # Extract OpenAI format parameters
        model = payload.get("model")
        messages = payload.get("messages", [])
        temperature = payload.get("temperature", DEFAULT_TEMPERATURE)
        max_tokens = payload.get("max_tokens", DEFAULT_MAX_NEW_TOKENS)
        top_p = payload.get("top_p", DEFAULT_TOP_P)
        stream = payload.get("stream", False)
        tools = payload.get("tools")
        
        # Convert to Ollama format
        manager = get_manager(request)
        session_mgr = request.app.state.session_manager
        
        # Ensure model is loaded
        ensure_model_loaded(manager, model, payload)
        
        # Convert messages to Ollama ChatMessage format
        from session_manager import ChatMessage
        chat_messages = []
        for msg in messages:
            chat_messages.append(ChatMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", "")
            ))
        
        # Build generation options
        options = {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "top_p": top_p
        }
        gen_options = parse_options(options)
        
        # Handle tools if present
        tool_manager = None
        if tools:
            tool_manager = ToolManager(tools=tools)
        
        # Generate response
        if stream:
            # Streaming response
            async def generate_stream():
                prompt = session_mgr.build_prompt(chat_messages, tool_manager)
                
                yield 'data: {"id":"chatcmpl-mi50","object":"chat.completion.chunk","created":0,"model":"' + model + '","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n'
                
                full_response = ""
                for chunk in manager.generate_stream(model, prompt, gen_options):
                    if chunk:
                        full_response += chunk
                        openai_chunk = {
                            "id": "chatcmpl-mi50",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(openai_chunk)}\n\n"
                
                # Final chunk
                final_chunk = {
                    "id": "chatcmpl-mi50",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        
        else:
            # Non-streaming response
            prompt = session_mgr.build_prompt(chat_messages, tool_manager)
            response_text = manager.generate_sync(model, prompt, gen_options)
            
            # Check for tool calls
            tool_calls_list = []
            if tool_manager:
                try:
                    tool_calls_list = tool_manager.parse_tool_calls(response_text)
                except (InvalidToolCallError, IncompleteToolCallError):
                    pass
            
            # Build OpenAI response
            message = {
                "role": "assistant",
                "content": response_text
            }
            
            if tool_calls_list:
                message["tool_calls"] = []
                for tc in tool_calls_list:
                    message["tool_calls"].append({
                        "id": f"call_{tc.get('id', 'unknown')}",
                        "type": "function",
                        "function": {
                            "name": tc.get("name"),
                            "arguments": json.dumps(tc.get("arguments", {}))
                        }
                    })
            
            return {
                "id": "chatcmpl-mi50",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }


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
