"""Utility helpers for the MI50 PyTorch Ollama-compatible service."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import torch

LOGGER = logging.getLogger("mi50_ollama")

_LOG_RECORD_SKIP_FIELDS = {
    "args",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class JsonLogFormatter(logging.Formatter):
    """Minimal JSON log formatter with timestamp and message."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: Dict[str, Any] = {
            "ts": now_iso(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info

        for key, value in record.__dict__.items():
            if key in _LOG_RECORD_SKIP_FIELDS or key.startswith("_"):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload.setdefault("extra", {})[key] = value
            else:
                payload.setdefault("extra", {})[key] = repr(value)

        return json.dumps(payload, ensure_ascii=False)


def _normalize_log_level(level: Union[str, int, None]) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        candidate = level.strip()
        if candidate.isdigit():
            return int(candidate)
        resolved = logging.getLevelName(candidate.upper())
        if isinstance(resolved, int):
            return resolved
    return logging.INFO


def configure_logging(
    *,
    level: Union[str, int, None] = None,
    log_dir: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Configure root logger with JSON formatter and optional rotating file handler."""

    logging.captureWarnings(True)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    resolved_level = _normalize_log_level(level)
    root.setLevel(resolved_level)
    formatter = JsonLogFormatter()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_dir:
        path = Path(log_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path / "mi50_ollama.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 (millisecond precision)."""

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def detect_device() -> Dict[str, Any]:
    """Inspect torch runtime and return metadata about the available accelerator."""

    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Unable to fetch GPU name: %s", exc)
            gpu_name = "CUDA GPU"
        device = "cuda"
    else:
        gpu_name = "CPU"
        device = "cpu"

    dtype = torch.bfloat16 if is_rocm else (torch.float16 if cuda_available else torch.float32)

    return {
        "device": device,
        "gpu_name": gpu_name,
        "is_rocm": is_rocm,
        "dtype": dtype,
    }


def apply_stop_sequences(text: str, stop: Optional[Sequence[str]]) -> str:
    """Trim *text* at the first occurrence of any sequence in *stop*."""

    if not stop:
        return text

    cut = len(text)
    for token in stop:
        if not token:
            continue
        idx = text.find(token)
        if idx != -1 and idx < cut:
            cut = idx
    return text[:cut]


def chunk_string(new_text: str, previous_text: str) -> str:
    """Return only the delta between *new_text* and *previous_text*."""

    if new_text.startswith(previous_text):
        return new_text[len(previous_text) :]
    return new_text


class OllamaError(Exception):
    """Domain specific error used by the service."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = message


def build_ndjson(payload: Dict[str, Any]) -> str:
    """Serialize payload as JSON line."""

    return json.dumps(payload, ensure_ascii=False) + "\n"
