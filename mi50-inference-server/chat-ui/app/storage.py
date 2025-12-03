from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .token_counter import count_messages_tokens, get_context_status, suggest_trim_strategy


def now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _default_title(model: str) -> str:
    name = model.split("/")[-1]
    return f"Sessione {name}"


def _default_system_message(model: str) -> str:
    """Messaggio di sistema predefinito"""
    return "Sei un assistente AI utile e competente. Rispondi in modo chiaro, preciso e conciso."


def _auto_tags(text: str, model: str) -> List[str]:
    tags = {model.split("/")[-1]}
    lowered = text.lower()
    keywords = {
        "esp32": "ESP32",
        "esp-idf": "ESP32",
        "arduino": "Arduino",
        "python": "Python",
        "raspberry": "Raspberry",
        "modbus": "Modbus",
        "termocoppia": "Sensori",
    }
    for key, label in keywords.items():
        if key in lowered:
            tags.add(label)
    if "http" in lowered or "web" in lowered:
        tags.add("Web")
    if any(tok in lowered for tok in ("codice", "code", "programma")):
        tags.add("Codice")
    return sorted(tags)


class ChatStorage:
    def __init__(self, path: Path, default_model: str) -> None:
        self.path = path
        self.default_model = default_model
        self.lock = threading.Lock()
        self._data: Dict[str, Any] = {"chats": []}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._data = {"chats": []}
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._data = {"chats": []}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def list_chats(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [
                {
                    "id": chat["id"],
                    "title": chat["title"],
                    "model": chat["model"],
                    "tags": chat.get("tags", []),
                    "system_message": chat.get("system_message", ""),
                    "updated_at": chat.get("updated_at", chat.get("created_at")),
                }
                for chat in self._data["chats"]
            ]

    def _find_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        for chat in self._data["chats"]:
            if chat["id"] == chat_id:
                return chat
        return None

    def get_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self._find_chat(chat_id)

    def create_chat(self, title: Optional[str] = None, model: Optional[str] = None, system_message: Optional[str] = None) -> Dict[str, Any]:
        model = model or self.default_model
        chat = {
            "id": uuid.uuid4().hex,
            "title": title or _default_title(model),
            "model": model,
            "system_message": system_message or _default_system_message(model),
            "tags": [],
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "messages": [],
        }
        with self.lock:
            self._data["chats"].insert(0, chat)
            self._save()
        return chat

    def append_message(self, chat_id: str, role: str, content: str) -> Dict[str, Any]:
        with self.lock:
            chat = self._find_chat(chat_id)
            if not chat:
                raise KeyError("Chat non trovata")
            entry = {"role": role, "content": content, "ts": now_iso()}
            chat["messages"].append(entry)
            chat["updated_at"] = now_iso()
            if role == "user":
                chat.setdefault("tags", [])
                new_tags = _auto_tags(content, chat["model"])
                chat["tags"] = sorted(set(chat["tags"]) | set(new_tags))
            self._save()
            return entry

    def update_title(self, chat_id: str, title: str) -> None:
        with self.lock:
            chat = self._find_chat(chat_id)
            if not chat:
                raise KeyError("Chat non trovata")
            chat["title"] = title
            chat["updated_at"] = now_iso()
            self._save()

    def update_system_message(self, chat_id: str, system_message: str) -> None:
        with self.lock:
            chat = self._find_chat(chat_id)
            if not chat:
                raise KeyError("Chat non trovata")
            chat["system_message"] = system_message
            chat["updated_at"] = now_iso()
            self._save()

    def ensure_chat(self, chat_id: str) -> Dict[str, Any]:
        chat = self._find_chat(chat_id)
        if not chat:
            raise KeyError("Chat non trovata")
        return chat

    def to_openai_messages(self, chat_id: str) -> List[Dict[str, str]]:
        chat = self.ensure_chat(chat_id)
        messages = []

        # Aggiungi il system message all'inizio se presente
        system_msg = chat.get("system_message", "")
        if system_msg:
            messages.append({"role": "system", "content": system_msg})

        # Aggiungi tutti i messaggi della conversazione
        for msg in chat.get("messages", []):
            messages.append({"role": msg["role"], "content": msg["content"]})

        return messages

    # ========== NUOVI METODI PER GESTIONE CONTEXT ==========

    def get_token_stats(self, chat_id: str) -> Dict[str, Any]:
        """
        Ottiene statistiche sui token della conversazione.

        Returns:
            Dict con conteggi token e stato del context
        """
        with self.lock:
            chat = self._find_chat(chat_id)
            if not chat:
                raise KeyError("Chat non trovata")

            messages = self.to_openai_messages(chat_id)
            model = chat["model"]

            # Conta token
            token_stats = count_messages_tokens(messages, model)

            # Valuta stato context
            context_status = get_context_status(token_stats["total"], model)

            # Suggerisci strategia se necessario
            suggestion = None
            if context_status["status"] in ("warning", "critical"):
                suggestion = suggest_trim_strategy(messages, model, target_percentage=0.5)

            return {
                "chat_id": chat_id,
                "model": model,
                "message_count": len(chat.get("messages", [])),
                "tokens": token_stats,
                "context": context_status,
                "suggestion": suggestion,
            }

    def trim_messages_sliding_window(self, chat_id: str, keep_last_n: int) -> int:
        """
        Taglia i messaggi mantenendo solo gli ultimi N (+ system message).

        Args:
            chat_id: ID della chat
            keep_last_n: Numero di messaggi da mantenere

        Returns:
            Numero di messaggi rimossi
        """
        with self.lock:
            chat = self._find_chat(chat_id)
            if not chat:
                raise KeyError("Chat non trovata")

            messages = chat.get("messages", [])
            original_count = len(messages)

            if original_count <= keep_last_n:
                return 0  # Nessun trimming necessario

            # Mantieni solo gli ultimi N messaggi
            removed_count = original_count - keep_last_n
            chat["messages"] = messages[-keep_last_n:]
            chat["updated_at"] = now_iso()

            self._save()
            return removed_count

    def trim_messages_to_target(self, chat_id: str, target_percentage: float = 0.5) -> Dict[str, Any]:
        """
        Taglia i messaggi fino a raggiungere una percentuale target del context.

        Args:
            chat_id: ID della chat
            target_percentage: Percentuale target (es: 0.5 = 50%)

        Returns:
            Dict con:
            - removed_count: Messaggi rimossi
            - kept_count: Messaggi mantenuti
            - tokens_before: Token prima del trim
            - tokens_after: Token dopo il trim
        """
        with self.lock:
            chat = self._find_chat(chat_id)
            if not chat:
                raise KeyError("Chat non trovata")

            model = chat["model"]
            messages_before = self.to_openai_messages(chat_id)

            # Ottieni suggerimento strategia
            suggestion = suggest_trim_strategy(messages_before, model, target_percentage)

            if suggestion["strategy"] == "none":
                return {
                    "removed_count": 0,
                    "kept_count": len(chat.get("messages", [])),
                    "tokens_before": suggestion["current_tokens"],
                    "tokens_after": suggestion["current_tokens"],
                }

            # Applica trim: mantieni ultimi keep_count messaggi
            keep_count = suggestion["keep_count"]
            messages = chat.get("messages", [])
            original_count = len(messages)

            if original_count <= keep_count:
                removed_count = 0
            else:
                removed_count = original_count - keep_count
                chat["messages"] = messages[-keep_count:]
                chat["updated_at"] = now_iso()
                self._save()

            # Ricalcola token dopo trim
            messages_after = self.to_openai_messages(chat_id)
            tokens_after = count_messages_tokens(messages_after, model)["total"]

            return {
                "removed_count": removed_count,
                "kept_count": keep_count,
                "tokens_before": suggestion["current_tokens"],
                "tokens_after": tokens_after,
            }

    def get_trimmed_messages_preview(self, chat_id: str, keep_last_n: int) -> List[Dict[str, str]]:
        """
        Ritorna un'anteprima dei messaggi dopo il trim, senza modificare lo storage.

        Args:
            chat_id: ID della chat
            keep_last_n: Numero di messaggi da mantenere

        Returns:
            Lista di messaggi in formato OpenAI (system + ultimi N)
        """
        with self.lock:
            chat = self._find_chat(chat_id)
            if not chat:
                raise KeyError("Chat non trovata")

            messages = []

            # Aggiungi system message se presente
            system_msg = chat.get("system_message", "")
            if system_msg:
                messages.append({"role": "system", "content": system_msg})

            # Aggiungi ultimi N messaggi conversazione
            chat_messages = chat.get("messages", [])
            for msg in chat_messages[-keep_last_n:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

            return messages
