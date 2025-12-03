"""Token counter per gestire context window dei modelli LLM."""
from __future__ import annotations

from typing import Dict, List, Optional
import re


# Limiti context window per modello (in token)
MODEL_LIMITS = {
    "qwen": 32768,      # Qwen 2.5 ha 32K context
    "gemma": 8192,      # Gemma 2 ha 8K context
    "deepseek": 16384,  # DeepSeek Coder ha 16K context
    "starcoder": 16384, # StarCoder2 ha 16K context
    "generic": 4096,    # Fallback conservativo
}

# Soglie per warning (percentuale del limite)
WARNING_THRESHOLD = 0.80  # Avvisa all'80%
CRITICAL_THRESHOLD = 0.95  # Critico al 95%


def detect_model_family(model_path: str) -> str:
    """Rileva la famiglia del modello dal path."""
    model_lower = model_path.lower()

    if "gemma" in model_lower:
        return "gemma"
    elif "qwen" in model_lower:
        return "qwen"
    elif "deepseek" in model_lower:
        return "deepseek"
    elif "starcoder" in model_lower or "star-coder" in model_lower:
        return "starcoder"
    else:
        return "generic"


def get_model_limit(model_path: str) -> int:
    """Ritorna il limite di token per il modello."""
    family = detect_model_family(model_path)
    return MODEL_LIMITS.get(family, MODEL_LIMITS["generic"])


def estimate_tokens_simple(text: str, model_family: str = "generic") -> int:
    """
    Stima il numero di token usando euristiche semplici.

    Regole approssimative:
    - 1 token ≈ 4 caratteri per testo inglese
    - 1 token ≈ 2-3 caratteri per codice (più denso)
    - 1 token ≈ 1.5 caratteri per testo italiano/multilingua

    Per maggiore precisione si dovrebbe usare tiktoken, ma richiede dipendenza extra.
    """
    if not text:
        return 0

    # Conta parole, numeri, punteggiatura
    words = len(re.findall(r'\w+', text))

    # Euristiche per famiglia di modello
    if model_family == "starcoder":
        # Codice: più denso, ~1 token per 3 caratteri
        return max(len(text) // 3, words)
    elif model_family == "gemma":
        # Gemma tende a tokenizzare in modo più granulare
        return int(words * 1.3)
    elif model_family == "qwen":
        # Qwen usa tokenizer simile a GPT (BPE)
        return int(words * 1.35)
    elif model_family == "deepseek":
        # DeepSeek simile a Qwen
        return int(words * 1.35)
    else:
        # Stima conservativa: 1.5 token per parola
        return int(words * 1.5)


def count_prompt_tokens(formatted_prompt: str, model_path: str) -> int:
    """
    Conta i token in un prompt già formattato.

    Args:
        formatted_prompt: Il prompt con i tag del modello (<|im_start|>, <start_of_turn>, etc.)
        model_path: Path del modello (es: "Qwen/Qwen2.5-7B-Instruct")

    Returns:
        Numero stimato di token
    """
    family = detect_model_family(model_path)

    # Aggiungi overhead per i tag speciali del modello
    base_tokens = estimate_tokens_simple(formatted_prompt, family)

    # Overhead tag: circa 2-5 token per messaggio
    if family == "qwen" or family == "deepseek":
        # <|im_start|>user\n ... <|im_end|> = ~4 token
        num_messages = formatted_prompt.count("<|im_start|>")
        overhead = num_messages * 4
    elif family == "gemma":
        # <start_of_turn>user\n ... <end_of_turn> = ~3 token
        num_messages = formatted_prompt.count("<start_of_turn>")
        overhead = num_messages * 3
    else:
        overhead = 10  # Stima conservativa

    return base_tokens + overhead


def count_messages_tokens(messages: List[Dict[str, str]], model_path: str) -> Dict[str, int]:
    """
    Conta i token di una lista di messaggi (formato OpenAI).

    Args:
        messages: Lista di dict con role/content
        model_path: Path del modello

    Returns:
        Dict con conteggi dettagliati:
        - total: Token totali
        - system: Token del system message
        - conversation: Token della conversazione (user + assistant)
        - per_message: Lista con token per messaggio
    """
    family = detect_model_family(model_path)

    total = 0
    system_tokens = 0
    conversation_tokens = 0
    per_message = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Stima token del contenuto
        msg_tokens = estimate_tokens_simple(content, family)

        # Aggiungi overhead per i tag
        if family in ("qwen", "deepseek"):
            msg_tokens += 4  # <|im_start|>{role}\n ... <|im_end|>
        elif family == "gemma":
            msg_tokens += 3  # <start_of_turn>{role}\n ... <end_of_turn>
        else:
            msg_tokens += 2  # Overhead generico

        per_message.append({
            "role": role,
            "tokens": msg_tokens,
            "content_preview": content[:50] + "..." if len(content) > 50 else content
        })

        total += msg_tokens

        if role == "system":
            system_tokens += msg_tokens
        else:
            conversation_tokens += msg_tokens

    # Aggiungi overhead per il prefisso della risposta
    if family in ("qwen", "deepseek"):
        total += 3  # <|im_start|>assistant\n
    elif family == "gemma":
        total += 2  # <start_of_turn>model\n

    return {
        "total": total,
        "system": system_tokens,
        "conversation": conversation_tokens,
        "per_message": per_message,
    }


def get_context_status(token_count: int, model_path: str) -> Dict[str, any]:
    """
    Valuta lo stato del contesto rispetto al limite del modello.

    Returns:
        Dict con:
        - used: Token usati
        - limit: Limite del modello
        - percentage: Percentuale usata (0-100)
        - remaining: Token rimanenti
        - status: "ok" | "warning" | "critical"
        - message: Messaggio descrittivo
    """
    limit = get_model_limit(model_path)
    percentage = (token_count / limit) * 100
    remaining = limit - token_count

    if percentage >= CRITICAL_THRESHOLD * 100:
        status = "critical"
        message = f"CRITICO: Context quasi pieno ({percentage:.1f}%). Rimuovi messaggi."
    elif percentage >= WARNING_THRESHOLD * 100:
        status = "warning"
        message = f"ATTENZIONE: Context all'{percentage:.1f}%. Considera di rimuovere messaggi."
    else:
        status = "ok"
        message = f"Context OK ({percentage:.1f}% usato)"

    return {
        "used": token_count,
        "limit": limit,
        "percentage": round(percentage, 1),
        "remaining": remaining,
        "status": status,
        "message": message,
    }


def suggest_trim_strategy(
    messages: List[Dict[str, str]],
    model_path: str,
    target_percentage: float = 0.5,
) -> Dict[str, any]:
    """
    Suggerisce una strategia di trimming per ridurre il context.

    Args:
        messages: Lista messaggi corrente
        model_path: Modello in uso
        target_percentage: Percentuale target (default 50%)

    Returns:
        Dict con:
        - current_tokens: Token attuali
        - target_tokens: Token target
        - strategy: Nome strategia consigliata
        - keep_count: Numero messaggi da mantenere
        - remove_count: Numero messaggi da rimuovere
        - description: Descrizione strategia
    """
    stats = count_messages_tokens(messages, model_path)
    limit = get_model_limit(model_path)
    target_tokens = int(limit * target_percentage)

    current = stats["total"]
    system_tokens = stats["system"]

    # Calcola quanti token dobbiamo risparmiare
    tokens_to_save = current - target_tokens

    if tokens_to_save <= 0:
        return {
            "current_tokens": current,
            "target_tokens": target_tokens,
            "strategy": "none",
            "keep_count": len(messages),
            "remove_count": 0,
            "description": "Nessun trimming necessario",
        }

    # Strategia: Sliding Window (mantieni ultimi N messaggi + system)
    # Rimuovi i messaggi più vecchi finché non raggiungiamo il target

    # Conta token accumulati dall'ultima risposta
    accumulated = system_tokens
    keep_count = 0

    # Scorri messaggi dal più recente al più vecchio (escludi system)
    conversation_messages = [m for m in messages if m["role"] != "system"]

    for msg_info in reversed(stats["per_message"]):
        if msg_info["role"] == "system":
            continue

        if accumulated + msg_info["tokens"] <= target_tokens:
            accumulated += msg_info["tokens"]
            keep_count += 1
        else:
            break

    # Assicurati di mantenere almeno gli ultimi 2 messaggi (1 user + 1 assistant)
    keep_count = max(keep_count, 2)

    # Conta messaggi da rimuovere (escludi system message)
    remove_count = len(conversation_messages) - keep_count

    strategy_name = "sliding_window"
    description = f"Mantieni system message + ultimi {keep_count} messaggi. Rimuovi {remove_count} messaggi vecchi."

    return {
        "current_tokens": current,
        "target_tokens": target_tokens,
        "strategy": strategy_name,
        "keep_count": keep_count,
        "remove_count": remove_count,
        "description": description,
    }


__all__ = [
    "get_model_limit",
    "estimate_tokens_simple",
    "count_prompt_tokens",
    "count_messages_tokens",
    "get_context_status",
    "suggest_trim_strategy",
]
