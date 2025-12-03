"""Simple in-memory session manager to persist chat history per session id."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str

    def normalised_role(self) -> str:
        return self.role.lower().strip()


@dataclass
class Session:
    session_id: str
    history: List[ChatMessage] = field(default_factory=list)

    def append(self, message: ChatMessage) -> None:
        self.history.append(message)

    def as_list(self) -> List[ChatMessage]:
        return list(self.history)


class SessionManager:
    """Thread-safe storage for chat sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(session_id=session_id)
            return self._sessions[session_id]

    def append(self, session_id: str, message: ChatMessage) -> None:
        session = self.get(session_id)
        session.append(message)

    def build_prompt(self, initial_messages: Sequence[ChatMessage], session_id: str | None) -> str:
        lines: List[str] = []

        for message in initial_messages:
            role = message.normalised_role()
            lines.append(f"{role.upper()}: {message.content}")

        if session_id:
            for past in self.get(session_id).as_list():
                role = past.normalised_role()
                lines.append(f"{role.upper()}: {past.content}")

        lines.append("ASSISTANT:")
        return "\n".join(lines)

    def save_exchange(
        self,
        session_id: str,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
    ) -> None:
        self.append(session_id, user_message)
        self.append(session_id, assistant_message)


__all__ = ["ChatMessage", "SessionManager", "Session"]
