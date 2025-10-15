"""Model loading and text generation helpers."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from utils import LOGGER, OllamaError, apply_stop_sequences, chunk_string, now_iso


@dataclass
class GenerationOptions:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repetition_penalty: float = 1.0
    seed: Optional[int] = None
    stop: Optional[List[str]] = None


@dataclass
class ModelHandle:
    name: str
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    dtype: torch.dtype
    device: str
    loaded_at: str = field(default_factory=now_iso)

    @property
    def context_length(self) -> int:
        return getattr(self.model.config, "max_position_embeddings", 4096)

    @property
    def num_parameters(self) -> int:
        try:
            return sum(p.numel() for p in self.model.parameters())
        except Exception:  # pragma: no cover - defensive
            return 0


class ModelManager:
    """Lazy loader for HuggingFace causal models."""

    def __init__(self, device: str, dtype: torch.dtype, quantization_allowed: bool = False) -> None:
        self.device = device
        self.dtype = dtype
        self.quantization_allowed = quantization_allowed
        self._models: Dict[str, ModelHandle] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # metadata helpers
    # ------------------------------------------------------------------
    def list_models(self) -> List[Dict[str, object]]:
        items: List[Dict[str, object]] = []
        for handle in self._models.values():
            items.append(
                {
                    "name": handle.name,
                    "size": handle.num_parameters,
                    "digest": handle.loaded_at,
                    "details": {
                        "context_length": handle.context_length,
                        "num_parameters": handle.num_parameters,
                        "dtype": str(handle.dtype),
                        "device": handle.device,
                    },
                }
            )
        return items

    def get_handle(self, model_name: str) -> ModelHandle:
        with self._lock:
            if model_name not in self._models:
                raise OllamaError(f"Model '{model_name}' is not loaded", status_code=404)
            return self._models[model_name]

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def load_model(self, model_name: str, quantize: Optional[str] = None) -> ModelHandle:
        if quantize and not self.quantization_allowed:
            raise OllamaError(
                "Quantisation requested but not supported on this backend (ROCm MI50)",
                status_code=400,
            )

        with self._lock:
            if model_name in self._models:
                LOGGER.info("Model %s already loaded", model_name)
                return self._models[model_name]

            LOGGER.info("Loading %s (dtype=%s, device=%s)", model_name, self.dtype, self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            if self.device != "cuda":
                model = model.to(self.device)
            model.eval()
            handle = ModelHandle(
                name=model_name,
                tokenizer=tokenizer,
                model=model,
                dtype=self.dtype,
                device=self.device,
            )
            self._models[model_name] = handle
            return handle

    def unload_model(self, model_name: str) -> bool:
        with self._lock:
            handle = self._models.pop(model_name, None)
        if handle:
            LOGGER.info("Unloaded model %s", model_name)
            try:
                del handle.model
            except Exception:  # pragma: no cover
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return False

    # ------------------------------------------------------------------
    # generation helpers
    # ------------------------------------------------------------------
    def _build_inputs(self, handle: ModelHandle, prompt: str):
        inputs = handle.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=handle.context_length,
        ).to(handle.model.device)
        return inputs

    def generate_sync(
        self,
        model_name: str,
        prompt: str,
        options: GenerationOptions,
    ) -> str:
        handle = self.get_handle(model_name)
        inputs = self._build_inputs(handle, prompt)
        kwargs = self._generation_kwargs(handle, options)
        output = handle.model.generate(**inputs, **kwargs)
        generated = output[0][inputs["input_ids"].shape[1] :]
        text = handle.tokenizer.decode(generated, skip_special_tokens=True)
        text = apply_stop_sequences(text, options.stop)
        return text

    def generate_stream(
        self,
        model_name: str,
        prompt: str,
        options: GenerationOptions,
    ) -> Iterator[str]:
        handle = self.get_handle(model_name)
        inputs = self._build_inputs(handle, prompt)
        streamer = TextIteratorStreamer(
            handle.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        kwargs = self._generation_kwargs(handle, options)
        kwargs["streamer"] = streamer

        thread = threading.Thread(target=handle.model.generate, kwargs={**inputs, **kwargs}, daemon=True)
        thread.start()

        full_text = ""
        for token_text in streamer:
            candidate = full_text + token_text
            truncated = apply_stop_sequences(candidate, options.stop)
            delta = chunk_string(truncated, full_text)
            if delta:
                yield delta
            full_text = truncated
            if truncated != candidate:
                break

        thread.join()
        yield ""  # signal completion

    def _generation_kwargs(self, handle: ModelHandle, options: GenerationOptions) -> Dict[str, object]:
        generator = None
        if options.seed is not None:
            generator = torch.Generator(device=handle.model.device)
            generator.manual_seed(int(options.seed))

        kwargs: Dict[str, object] = {
            "max_new_tokens": options.max_new_tokens,
            "temperature": max(options.temperature, 1e-5),
            "top_p": options.top_p,
            "repetition_penalty": options.repetition_penalty,
            "do_sample": options.temperature > 0.0,
            "pad_token_id": handle.tokenizer.pad_token_id,
            "eos_token_id": handle.tokenizer.eos_token_id,
        }
        if options.top_k and options.top_k > 0:
            kwargs["top_k"] = options.top_k
        if generator is not None:
            kwargs["generator"] = generator
        return kwargs


def parse_options(payload: Optional[Dict[str, object]] = None) -> GenerationOptions:
    payload = payload or {}
    max_new_tokens = int(payload.get("num_predict", payload.get("max_new_tokens", 256)))
    if max_new_tokens <= 0:
        max_new_tokens = 256
    temperature = float(payload.get("temperature", 0.7))
    top_p = float(payload.get("top_p", 0.95))
    top_k = int(payload.get("top_k", 40))
    repetition_penalty = float(payload.get("repeat_penalty", payload.get("repetition_penalty", 1.0)))
    seed = payload.get("seed")
    stop = payload.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    elif stop is None:
        stop = None
    elif isinstance(stop, Iterable):
        stop = [str(item) for item in stop]
    else:
        stop = None
    return GenerationOptions(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        seed=int(seed) if seed is not None else None,
        stop=stop,
    )


__all__ = ["ModelManager", "GenerationOptions", "parse_options"]
