from __future__ import annotations

import asyncio
from typing import Any, Dict, Set


class TokenBroadcaster:
    """In-process pub/sub for streaming token events."""

    def __init__(self) -> None:
        self._subscribers: Set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def publish(self, event: Dict[str, Any]) -> None:
        if not self._loop:
            return
        self._loop.call_soon_threadsafe(self._dispatch, event)

    def _dispatch(self, event: Dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                queue.put_nowait(event)
