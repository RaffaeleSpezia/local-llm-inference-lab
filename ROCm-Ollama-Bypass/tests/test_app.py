from __future__ import annotations

import json
from typing import Dict, Iterator, List, Optional

import pytest
from fastapi.testclient import TestClient

import app as app_module
from session_manager import ChatMessage


class DummyHandle:
    def __init__(self, name: str = "dummy"):
        self.name = name
        self.context_length = 4096
        self.num_parameters = 42
        self.dtype = "float32"
        self.device = "cpu"


class DummyModelManager:
    def __init__(self) -> None:
        self.loaded: List[str] = []
        self.handle = DummyHandle()

    def load_model(self, model_name: str, quantize: Optional[str] = None):
        if quantize:
            raise ValueError("quantize not supported")
        self.loaded.append(model_name)
        return self.handle

    def list_models(self):
        return [{"name": self.handle.name, "size": 42, "digest": "x"}]

    def get_handle(self, model_name: str):
        return self.handle

    def unload_model(self, model_name: str) -> bool:
        return True

    def generate_sync(self, model_name: str, prompt: str, options):
        return f"response to {prompt}"

    def generate_stream(self, model_name: str, prompt: str, options) -> Iterator[str]:
        yield "hello "
        yield "world"
        yield ""


class DummyRAGStore:
    def upsert(self, dataset_id: str, items):
        return len(items)

    def build_augmented_prompt(self, dataset_id: str, query: str, top_k: int = 3):
        return {
            "chunks": [{"id": "1", "text": "chunk", "score": 0.9}],
            "augmented_prompt": f"context: {query}",
        }


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    application = app_module.create_app()
    application.state.model_manager = DummyModelManager()
    application.state.rag_store = DummyRAGStore()
    return TestClient(application)


def test_version(client: TestClient):
    resp = client.get("/api/version")
    data = resp.json()
    assert resp.status_code == 200
    assert "gpu" in data


def test_generate_sync(client: TestClient):
    payload = {"prompt": "hello"}
    resp = client.post("/api/generate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["done"] is True
    assert "response" in data


def test_generate_stream(client: TestClient):
    payload = {"prompt": "hello", "stream": True}
    resp = client.post("/api/generate", json=payload)
    assert resp.status_code == 200
    lines = [json.loads(line) for line in resp.text.strip().split("\n")]
    assert len(lines) >= 2
    assert lines[-1]["done"] is True
    assert any((not chunk["done"]) for chunk in lines[:-1])


def test_chat_session(client: TestClient):
    payload = {
        "messages": [
            {"role": "user", "content": "Ciao"},
        ]
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["done"] is True
    assert "session_id" in body


def test_rag_endpoints(client: TestClient):
    upsert_payload = {
        "dataset_id": "docs",
        "documents": [
            {"id": "a", "text": "alpha"},
            {"id": "b", "text": "beta"},
        ],
    }
    resp = client.post("/rag/upsert", json=upsert_payload)
    assert resp.status_code == 200
    resp = client.post(
        "/rag/query",
        json={"dataset_id": "docs", "query": "ciao", "top_k": 1},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "augmented_prompt" in body
