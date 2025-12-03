#!/usr/bin/env python3
"""Interactive chat helper for the MI50 Ollama-like service."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests

SYSTEM_CONTENT = (
    "Sei Qwen2.5-Coder su GPU AMD MI50. Rispondi in italiano con codice chiaro, "
    "commenti essenziali e indicazioni pratiche."
)
SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    f"{SYSTEM_CONTENT}\n"
    "<|im_end|>\n"
)
USER_TAG = "<|im_start|>user\n{}\n<|im_end|>\n"
ASSISTANT_TAG = "<|im_start|>assistant\n"
ASSISTANT_CLOSE = "\n<|im_end|>\n"
STOP_SEQUENCES: List[str] = ["<|im_end|>", "<|im_start|>user"]

ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"

DEFAULT_HISTORY_TEMPLATE = [
    {
        "role": "user",
        "content": "Prima di iniziare ricordami in breve come prepari il contesto e come presenterai il codice (usa Markdown).",
    },
    {
        "role": "assistant",
        "content": "Certo! Imposto sempre il contesto Qwen con i tag <|im_start|>/<|im_end|>, rispondo in italiano e presento il codice in blocchi Markdown. Pronto per il tuo prompt.",
    },
]
DEFAULT_TEMPLATE_PATH = Path(__file__).with_name("chat_mi50_default.json")


def supports_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    return os.isatty(1)

COLOR_ENABLED = supports_color()


def color_text(text: str, color: str) -> str:
    if COLOR_ENABLED:
        return f"{color}{text}{ANSI_RESET}"
    return text


def build_conversation(history: List[Dict[str, str]]) -> str:
    convo = SYSTEM_PROMPT
    for message in history:
        if message["role"] == "user":
            convo += USER_TAG.format(message["content"].strip())
            convo += ASSISTANT_TAG
        else:
            convo += message["content"].rstrip("\n") + ASSISTANT_CLOSE
    return convo


def open_in_editor(conversation: str) -> str:
    editor_cmd = os.environ.get("CHAT_MI50_EDITOR")
    cmd = shlex.split(editor_cmd) if editor_cmd else ["nano", "--softwrap"]

    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(conversation)
        tmp.flush()
    try:
        subprocess.run(cmd + [str(tmp_path)], check=False)
        content = tmp_path.read_text(encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)

    if content.startswith(conversation):
        user_part = content[len(conversation) :]
    else:
        user_part = content
    return user_part.strip()


def call_service(host: str, prompt: str, model: str, num_predict: int, stop: List[str]) -> dict:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "prompt": prompt,
        "model": model,
        "stream": False,
        "options": {"stop": stop, "num_predict": num_predict},
    }
    response = requests.post(url, json=payload, timeout=1200)
    response.raise_for_status()
    return response.json()


def detect_stop_reason(text: str, stop_sequences: List[str]) -> str:
    if text.endswith("<|im_end|>"):
        return "tag <|im_end|>"
    for stop in stop_sequences:
        if stop and text.endswith(stop):
            return stop
    return "EOS o limite token"


def load_history(path: Path) -> List[Dict[str, str]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def save_history(path: Path, history: List[Dict[str, str]]) -> None:
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def load_default_history() -> List[Dict[str, str]]:
    if DEFAULT_TEMPLATE_PATH.exists():
        try:
            return json.loads(DEFAULT_TEMPLATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return DEFAULT_HISTORY_TEMPLATE.copy()


def print_prompt_view(history: List[Dict[str, str]]) -> None:
    print("system:\n" + SYSTEM_CONTENT)
    for message in history:
        role = message["role"]
        content = message["content"]
        color = ANSI_GREEN if role == "user" else ANSI_YELLOW
        print()
        print(color_text(f"{role}:\n{content}", color))


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive nano-based chat for MI50 service")
    parser.add_argument("command", nargs="?", choices=["new"], help="'new' per usare il contesto predefinito")
    parser.add_argument("--host", default="http://127.0.0.1:11534", help="Base URL del servizio")
    parser.add_argument(
        "--model",
        default="/mnt/raid0/qwen2.5-coder-7b-instruct",
        help="Percorso/nome modello da usare",
    )
    parser.add_argument(
        "--history",
        default="chat_mi50_history.json",
        help="File dove salvare il contesto (JSON con messaggi)",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=10000,
        help="Numero di token da generare per default",
    )
    args = parser.parse_args()

    history_path = Path(args.history)

    if args.command == "new":
        history = load_default_history()
        save_history(history_path, history)
        print(f"Context di default copiato in {history_path}.")
    else:
        history = load_history(history_path)

    print("Chat MI50 avviata. Il buffer mostra il prompt completo con i tag; scrivi in coda. Vuoto = esci.\n")

    while True:
        conversation = build_conversation(history)
        print("===== CONTESTO ATTUALE (FORMATTATO) =====")
        print_prompt_view(history)
        print("========================================\n")

        user_input = open_in_editor(conversation)
        if not user_input:
            print("Nessun input: chat terminata.")
            break

        history.append({"role": "user", "content": user_input})
        prompt = build_conversation(history)

        try:
            result = call_service(
                host=args.host,
                prompt=prompt,
                model=args.model,
                num_predict=args.num_predict,
                stop=STOP_SEQUENCES,
            )
        except requests.RequestException as exc:
            print(f"Errore nella chiamata al servizio: {exc}")
            history.pop()
            break

        answer_raw = result.get("response", "").strip()
        history.append({"role": "assistant", "content": answer_raw})
        save_history(history_path, history)

        stop_reason = detect_stop_reason(answer_raw, STOP_SEQUENCES)
        clean_answer = answer_raw.replace("<|im_start|>assistant\n", "").replace("<|im_end|>", "").strip()
        print("\n===== RISPOSTA MODELLO =====\n")
        print(color_text(clean_answer, ANSI_YELLOW))
        print(f"\n[Fine output rilevata: {stop_reason}]\n")
        print("============================\n")

    if history:
        suffix = timestamp()
        backup = history_path.with_name(f"{history_path.stem}_{suffix}{history_path.suffix}")
        save_history(backup, history)
        print(f"Conversazione salvata in {history_path} e backup {backup}")


if __name__ == "__main__":
    main()
