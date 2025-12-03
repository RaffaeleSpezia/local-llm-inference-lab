#!/usr/bin/env python3
"""Test del formatter per vedere il prompt generato"""
import sys
sys.path.insert(0, "/home/lele/mi50_stack/mi50_chat_ui")

from app.prompt_formatter import format_prompt_for_model

# Test Gemma3 con system message
messages_gemma = [
    {"role": "system", "content": "Sei un assistente tecnico. Rispondi sempre in italiano in modo breve e preciso."},
    {"role": "user", "content": "Che modello sei?"}
]

print("=== GEMMA3 PROMPT ===")
prompt = format_prompt_for_model("/mnt/raid0/gemma-7b", messages_gemma)
print(prompt)
print("=== END ===\n")

# Test Qwen
messages_qwen = [
    {"role": "system", "content": "Sei un assistente tecnico per ESP32."},
    {"role": "user", "content": "Come funziona I2C?"}
]

print("=== QWEN PROMPT ===")
prompt = format_prompt_for_model("/mnt/raid0/qwen2.5-coder-7b-instruct", messages_qwen)
print(prompt)
print("=== END ===")
