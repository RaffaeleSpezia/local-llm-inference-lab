"""Minimal retrieval augmented generation utilities."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from utils import LOGGER


class RAGEmbedder:
    """Generate sentence embeddings using a transformer encoder."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Loading RAG embedder %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed(self, text: str) -> List[float]:
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return embeddings.cpu().numpy().astype(np.float32).tolist()


class RAGStore:
    """Persist text chunks and embeddings for lightweight vector search."""

    def __init__(self, storage_dir: Path, embedder: Optional[RAGEmbedder] = None) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or RAGEmbedder(device="cpu")
        self._lock = threading.Lock()

    def _dataset_path(self, dataset_id: str) -> Path:
        return self.storage_dir / f"{dataset_id}.json"

    def _load_dataset(self, dataset_id: str) -> Dict[str, Any]:
        path = self._dataset_path(dataset_id)
        if not path.exists():
            return {"items": []}
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save_dataset(self, dataset_id: str, content: Dict[str, Any]) -> None:
        path = self._dataset_path(dataset_id)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(content, fh, ensure_ascii=False, indent=2)

    def upsert(self, dataset_id: str, items: Iterable[Dict[str, Any]]) -> int:
        with self._lock:
            dataset = self._load_dataset(dataset_id)
            existing = {entry["id"]: entry for entry in dataset.get("items", [])}
            count = 0
            for item in items:
                text = item.get("text")
                if not text:
                    continue
                chunk_id = item.get("id") or f"chunk-{len(existing)+1}"
                embedding = self.embedder.embed(text)
                payload = {
                    "id": chunk_id,
                    "text": text,
                    "embedding": embedding,
                    "metadata": item.get("metadata", {}),
                }
                existing[chunk_id] = payload
                count += 1
            dataset["items"] = list(existing.values())
            self._save_dataset(dataset_id, dataset)
            return count

    def query(self, dataset_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        with self._lock:
            dataset = self._load_dataset(dataset_id)
        items = dataset.get("items", [])
        if not items:
            return []

        query_vec = np.array(self.embedder.embed(query))
        doc_vectors = np.array([item["embedding"] for item in items])
        # Cosine similarity
        query_norm = np.linalg.norm(query_vec) + 1e-8
        doc_norm = np.linalg.norm(doc_vectors, axis=1) + 1e-8
        scores = (doc_vectors @ query_vec) / (doc_norm * query_norm)
        top_indices = scores.argsort()[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            item = items[int(idx)]
            results.append({
                "id": item["id"],
                "text": item["text"],
                "score": float(scores[int(idx)]),
                "metadata": item.get("metadata", {}),
            })
        return results

    def build_augmented_prompt(
        self,
        dataset_id: str,
        query: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        chunks = self.query(dataset_id, query, top_k=top_k)
        context = "\n\n".join(f"[{c['score']:.2f}] {c['text']}" for c in chunks)
        augmented_prompt = (
            "You are given additional context blocks. Use them to answer the user request.\n"
            "Context:\n"
            f"{context}\n\n"
            "User request:\n"
            f"{query}\n"
        )
        return {"chunks": chunks, "augmented_prompt": augmented_prompt}


__all__ = ["RAGStore", "RAGEmbedder"]
