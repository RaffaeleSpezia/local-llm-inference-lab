"""Utility per gestire tool calling su MI50."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class IncompleteToolCallError(Exception):
    """Segnala che il JSON non è ancora completo."""


class InvalidToolCallError(Exception):
    """Segnala un payload tool_call con struttura non valida."""


@dataclass
class ToolCall:
    """Rappresentazione normalizzata di una tool call."""

    id: str
    name: str
    arguments: str

    def to_openai_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


class ToolManager:
    """Gestisce definizioni tool e parsing delle risposte del modello."""

    def __init__(self, tools: Optional[List[Dict[str, Any]]] = None) -> None:
        self._tools = tools or []
        self._system_prompt: Optional[str] = None

    @property
    def tools(self) -> List[Dict[str, Any]]:
        return self._tools

    def has_tools(self) -> bool:
        return bool(self._tools)

    def build_system_prompt(self) -> str:
        if self._system_prompt is None:
            tools_json = json.dumps(self._tools, ensure_ascii=False, indent=2)
            self._system_prompt = (
                "You are a precise coding assistant with access to external tools.\n"
                "The tools follow the OpenAI tool specification and are provided as JSON.\n"
                "TOOLS:\n"
                f"{tools_json}\n\n"
                "If a tool is required respond ONLY with a JSON object (no markdown) matching:\n"
                "{\"tool_calls\": [{\"name\": \"<tool_name>\", \"arguments\": { ... }}]}\n"
                "For multiple tools include multiple entries in the list. If no tool is needed respond in plain text."
            )
        return self._system_prompt

    def parse_tool_calls(self, text: str) -> List[ToolCall]:
        stripped = text.strip()
        if not stripped:
            raise IncompleteToolCallError
        if not stripped.startswith("{") and not stripped.startswith("["):
            raise InvalidToolCallError

        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise IncompleteToolCallError from exc

        entries: List[Dict[str, Any]]
        if isinstance(payload, dict):
            if "tool_calls" in payload and isinstance(payload["tool_calls"], list):
                entries = payload["tool_calls"]
            elif {"name", "arguments"}.issubset(payload.keys()):
                entries = [payload]
            elif (
                "function" in payload
                and isinstance(payload["function"], dict)
                and {"name", "arguments"}.issubset(payload["function"].keys())
            ):
                entries = [payload]
            else:
                raise InvalidToolCallError
        elif isinstance(payload, list):
            entries = payload
        else:
            raise InvalidToolCallError

        calls: List[ToolCall] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise InvalidToolCallError
            function_block = entry.get("function") if isinstance(entry.get("function"), dict) else None
            name = entry.get("name")
            arguments: Any = entry.get("arguments")
            if function_block:
                name = function_block.get("name", name)
                arguments = function_block.get("arguments", arguments)
            if not name:
                raise InvalidToolCallError

            call_id = entry.get("id") or f"call_{uuid.uuid4().hex[:8]}"

            if isinstance(arguments, (dict, list)):
                args_str = json.dumps(arguments, ensure_ascii=False)
            else:
                args_str = str(arguments) if arguments is not None else "{}"

            calls.append(ToolCall(id=str(call_id), name=str(name), arguments=args_str))

        if not calls:
            raise InvalidToolCallError

        return calls

    def to_openai_calls(self, calls: List[ToolCall]) -> List[Dict[str, Any]]:
        return [call.to_openai_dict() for call in calls]


__all__ = [
    "ToolManager",
    "ToolCall",
    "IncompleteToolCallError",
    "InvalidToolCallError",
]
