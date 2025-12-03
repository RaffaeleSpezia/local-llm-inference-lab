from __future__ import annotations

import asyncio
import re
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Set

import httpx
import psutil


class MetricsPoller:
    def __init__(self, base_url: str, interval: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.interval = interval
        self._task: asyncio.Task | None = None
        self._subscribers: Set[asyncio.Queue] = set()
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        timeout = httpx.Timeout(5.0, read=5.0, write=5.0)
        self._client = httpx.AsyncClient(timeout=timeout)
        self._task = asyncio.create_task(self._run())

    async def shutdown(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._client:
            await self._client.aclose()
            self._client = None
        for queue in list(self._subscribers):
            queue.put_nowait({"type": "metrics", "payload": None})
            self._subscribers.remove(queue)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=5)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    async def _run(self) -> None:
        assert self._client is not None
        while True:
            payload = await self._collect(self._client)
            for queue in list(self._subscribers):
                try:
                    queue.put_nowait(payload)
                except asyncio.QueueFull:
                    queue.get_nowait()
                    queue.put_nowait(payload)
            await asyncio.sleep(self.interval)

    async def _collect(self, client: httpx.AsyncClient) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        cpu_percent = psutil.cpu_percent(interval=None)
        ram_percent = psutil.virtual_memory().percent
        memory_stats: Dict[str, Any] | None = None
        gpu_stats = self._gpu_metrics()
        try:
            resp = await client.get(f"{self.base_url}/debug/memory")
            resp.raise_for_status()
            data = resp.json()
            memory_stats = data.get("memory")
        except Exception:
            memory_stats = None

        return {
            "type": "metrics",
            "payload": {
                "ts": now,
                "cpu_percent": cpu_percent,
                "ram_percent": ram_percent,
                "memory": memory_stats,
                "gpu": gpu_stats,
            },
        }

    def _gpu_metrics(self) -> Dict[str, Any] | None:
        try:
            result = subprocess.run(
                [
                    "rocm-smi",
                    "--showtemp",
                    "--showuse",
                    "--showmeminfo",
                    "vram",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None

        output = result.stdout
        temp = self._match_float(r"Temperature\s*\(Sensor\s*edge\)\s*\(C\):\s*([0-9.]+)", output)
        if temp is None:
            temp = self._match_float(r"Temperature\s*\(Sensor\s*0\)\s*:\s*([0-9.]+)", output)
        util = self._match_float(r"GPU\s+use\s*\(\%\)\s*:\s*([0-9.]+)", output)
        total_b = self._match_int(r"VRAM\s+Total\s+Memory\s*\(B\)\s*:\s*([0-9]+)", output) or self._match_int(r"Total\s+Memory\s*\(B\)\s*:\s*([0-9]+)", output)
        used_b = self._match_int(r"VRAM\s+Total\s+Used\s+Memory\s*\(B\)\s*:\s*([0-9]+)", output) or self._match_int(r"Used\s+Memory\s*\(B\)\s*:\s*([0-9]+)", output)
        free_b = self._match_int(r"Free\s+Memory\s*\(B\)\s*:\s*([0-9]+)", output)
        if free_b is None and total_b is not None and used_b is not None:
            free_b = max(total_b - used_b, 0)

        def to_gb(value: int | None) -> float | None:
            if value is None:
                return None
            return value / (1024 ** 3)

        return {
            "temperature_c": temp,
            "utilization_percent": util,
            "vram_total_gb": to_gb(total_b),
            "vram_used_gb": to_gb(used_b),
            "vram_free_gb": to_gb(free_b),
        }

    @staticmethod
    def _match_float(pattern: str, text: str) -> float | None:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    @staticmethod
    def _match_int(pattern: str, text: str) -> int | None:
        match = re.search(pattern, text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None
