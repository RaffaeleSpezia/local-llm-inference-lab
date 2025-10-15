"""Simple smoke test for the MI50 Ollama-compatible server."""
from __future__ import annotations

import argparse
import time
from typing import Any, Dict

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for Ollama-compatible server")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--prompt", default="Say hello in Italian.")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    url = f"{args.host.rstrip('/')}/api/generate"
    payload: Dict[str, Any] = {"prompt": args.prompt, "stream": args.stream}

    start = time.time()
    if args.stream:
        with requests.post(url, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    print(line)
    else:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        print(resp.json())
    duration = time.time() - start
    print(f"Completed in {duration:.2f}s")


if __name__ == "__main__":
    main()
