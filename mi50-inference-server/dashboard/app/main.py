from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .metrics import MetricsPoller
from .log_watcher import TokenRelay

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
INDEX_HTML = STATIC_DIR / "index.html"

INFERENCE_URL = os.environ.get("MI50_SERVER_URL", "http://127.0.0.1:11534").rstrip("/")

app = FastAPI(title="MI50 Dashboard", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

metrics_poller = MetricsPoller(INFERENCE_URL)
token_relay = TokenRelay(INFERENCE_URL)


@app.on_event("startup")
async def on_startup() -> None:
    await metrics_poller.start()
    await token_relay.start()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await metrics_poller.shutdown()
    await token_relay.shutdown()


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


@app.websocket("/ws/metrics")
async def metrics_socket(ws: WebSocket) -> None:
    await ws.accept()
    queue = metrics_poller.subscribe()
    try:
        while True:
            payload = await queue.get()
            if payload.get("payload") is None:
                break
            await ws.send_json(payload)
    except WebSocketDisconnect:
        pass
    finally:
        metrics_poller.unsubscribe(queue)


@app.websocket("/ws/prompts")
async def prompt_socket(ws: WebSocket) -> None:
    await ws.accept()
    queue = token_relay.subscribe()
    try:
        while True:
            event = await queue.get()
            await ws.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        token_relay.unsubscribe(queue)
