from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Set
from urllib.parse import urlparse, urlunparse

import websockets


def _http_to_ws(url: str) -> str:
    parsed = urlparse(url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse(parsed._replace(scheme=scheme))


class TokenRelay:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        ws_base = _http_to_ws(self.base_url)
        self.ws_url = f"{ws_base}/ws/tokens"
        self._task: asyncio.Task | None = None
        self._subscribers: Set[asyncio.Queue] = set()

    async def start(self) -> None:
        if self._task:
            return
        self._task = asyncio.create_task(self._run())

    async def shutdown(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        for queue in list(self._subscribers):
            queue.put_nowait({"type": "tokens", "payload": None})
            self._subscribers.remove(queue)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    async def _run(self) -> None:
        while True:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
                    async for message in ws:
                        if not message:
                            continue
                        try:
                            payload = json.loads(message)
                        except json.JSONDecodeError:
                            continue
                        self._broadcast(payload)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(2.0)

    def _broadcast(self, payload: Dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                queue.put_nowait(payload)
