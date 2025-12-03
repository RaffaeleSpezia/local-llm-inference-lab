"""Model loading and text generation helpers."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from utils import LOGGER, OllamaError, apply_stop_sequences, chunk_string, now_iso

torch.set_grad_enabled(False)


def _env_int(name: str, default: int, minimum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(value, minimum)


def _env_float(name: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value


DEFAULT_MAX_NEW_TOKENS = _env_int("OLLAMA_FAKE_DEFAULT_MAX_NEW_TOKENS", 128, 1)
DEFAULT_TEMPERATURE = _env_float("OLLAMA_FAKE_DEFAULT_TEMPERATURE", 0.0, 0.0)
DEFAULT_TOP_P = _env_float("OLLAMA_FAKE_DEFAULT_TOP_P", 0.95, 0.0, 1.0)
DEFAULT_TOP_K = _env_int("OLLAMA_FAKE_DEFAULT_TOP_K", 0, 0)
DEFAULT_REPETITION_PENALTY = _env_float("OLLAMA_FAKE_DEFAULT_REPETITION_PENALTY", 1.0, 0.0)


@dataclass
class GenerationOptions:
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
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
        self.max_prompt_tokens = _env_int("OLLAMA_FAKE_MAX_PROMPT_TOKENS", 4096, 1)
        self.stream_chunk_chars = _env_int("OLLAMA_FAKE_STREAM_CHARS", 160, 0)
        self.attn_implementation = os.environ.get("OLLAMA_FAKE_ATTN_IMPL", "sdpa")

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

            # Se ci sono altri modelli caricati, scaricali PRIMA di caricare il nuovo
            # (MI50 ha solo 32GB VRAM - un modello alla volta)
            if self._models:
                LOGGER.info("Unloading previous models to free VRAM for %s", model_name)
                old_models = list(self._models.keys())
                for old_model in old_models:
                    handle = self._models.pop(old_model)
                    LOGGER.info("Unloading %s", old_model)
                    try:
                        del handle.model
                        del handle.tokenizer
                    except Exception as e:
                        LOGGER.warning("Error deleting model %s: %s", old_model, e)
                
                # Libera VRAM dopo unload
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    LOGGER.info("VRAM freed after unloading previous models")

            LOGGER.info("Loading %s (dtype=%s, device=%s)", model_name, self.dtype, self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            target_device = self.device if ":" in self.device else f"{self.device}:0"
            
            # Optimized loading for servers with abundant RAM (168GB+)
            # Strategy: Use RAM as temporary buffer to avoid VRAM double-allocation
            # Background: low_cpu_mem_usage=True + device_map causes PyTorch to allocate
            # temporary VRAM buffers (14GB model + 14GB buffers = 28GB waste)
            # With 168GB RAM available, we load to RAM first, then transfer to VRAM directly
            LOGGER.info("Loading model to RAM first (server has 168GB RAM available)")
            
            model_kwargs: Dict[str, object] = {
                "torch_dtype": self.dtype,
                # NO low_cpu_mem_usage - we have plenty of RAM, use it as buffer
                # NO device_map - direct transfer is faster and cleaner
            }
            if self.attn_implementation:
                model_kwargs["attn_implementation"] = self.attn_implementation

            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            except TypeError:
                # Fallback if attn_implementation not supported
                model_kwargs.pop("attn_implementation", None)
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

            # Transfer to GPU (works for both cuda and cpu)
            LOGGER.info("Transferring model from RAM to %s", target_device)
            model = model.to(target_device)
            
            # Critical: Free temporary buffers allocated during transfer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                LOGGER.info("Freed temporary VRAM buffers after model transfer")
            
            if getattr(model.config, "use_cache", None) is False:
                model.config.use_cache = True
            model.eval()

            # Forza garbage collection dopo eval per ridurre VRAM riservata
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                LOGGER.info("Freed reserved VRAM after model.eval()")
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
        max_length = min(self.max_prompt_tokens, handle.context_length)
        inputs = handle.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(handle.model.device)
        return inputs



    def generate_sync(
        self,
        model_name: str,
        prompt: str,
        options: GenerationOptions,
    ) -> str:
        handle = self.get_handle(model_name)
        if options.seed is not None:
            seed = int(options.seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
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
        if options.seed is not None:
            seed = int(options.seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
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
        buffer = ""
        for token_text in streamer:
            candidate = full_text + token_text
            truncated = apply_stop_sequences(candidate, options.stop)
            delta = chunk_string(truncated, full_text)
            full_text = truncated
            end_of_stream = truncated != candidate
            if delta:
                buffer += delta
                should_flush = (
                    self.stream_chunk_chars == 0
                    or "\n" in buffer
                    or len(buffer) >= self.stream_chunk_chars
                    or end_of_stream
                )
                if should_flush:
                    yield buffer
                    buffer = ""
            if end_of_stream:
                break

        thread.join()
        if buffer:
            yield buffer
        yield ""  # signal completion

        # Libera memoria GPU dopo generazione
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generation_kwargs(self, handle: ModelHandle, options: GenerationOptions) -> Dict[str, object]:
        use_sampling = options.temperature > 0.0
        kwargs: Dict[str, object] = {
            "max_new_tokens": options.max_new_tokens,
            "repetition_penalty": options.repetition_penalty,
            "do_sample": use_sampling,
            "pad_token_id": handle.tokenizer.pad_token_id,
            "eos_token_id": handle.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if use_sampling:
            kwargs["temperature"] = options.temperature
            kwargs["top_p"] = options.top_p
            if options.top_k and options.top_k > 0:
                kwargs["top_k"] = options.top_k
        else:
            kwargs["temperature"] = 0.0
        return kwargs





def parse_options(payload: Optional[Dict[str, object]] = None) -> GenerationOptions:
    payload = payload or {}

    def _read_int(key: str, fallback: int) -> int:
        value = payload.get(key, fallback)
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _read_float(key: str, fallback: float) -> float:
        value = payload.get(key, fallback)
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    max_new_tokens = _read_int("num_predict", _read_int("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    if max_new_tokens <= 0:
        max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    temperature = _read_float("temperature", DEFAULT_TEMPERATURE)
    if temperature < 0.0:
        temperature = DEFAULT_TEMPERATURE

    top_p = _read_float("top_p", DEFAULT_TOP_P)
    if top_p <= 0.0:
        top_p = DEFAULT_TOP_P
    elif top_p > 1.0:
        top_p = 1.0

    top_k = _read_int("top_k", DEFAULT_TOP_K)
    if top_k < 0:
        top_k = DEFAULT_TOP_K

    repetition_penalty = _read_float("repeat_penalty", _read_float("repetition_penalty", DEFAULT_REPETITION_PENALTY))
    if repetition_penalty <= 0.0:
        repetition_penalty = DEFAULT_REPETITION_PENALTY

    seed = payload.get("seed")
    stop = payload.get("stop")
    if isinstance(stop, str):
        stop = [stop]
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


__all__ = ["ModelManager", "GenerationOptions", "parse_options", "DEFAULT_MAX_NEW_TOKENS", "DEFAULT_TEMPERATURE", "DEFAULT_TOP_P"]
