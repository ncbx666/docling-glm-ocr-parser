from __future__ import annotations

import logging
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path

from .models import OcrResult, ParseSettings, TargetRegion

LOGGER = logging.getLogger(__name__)

PROMPTS = {
    "text": "Text Recognition:",
    "image": "Text Recognition:",
    "diagram": "Text Recognition:",
    "page": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}


class GlmOcrBackend(ABC):
    name: str

    @abstractmethod
    def recognize(self, target: TargetRegion) -> OcrResult:
        raise NotImplementedError


class DisabledGlmOcrBackend(GlmOcrBackend):
    name = "none"

    def recognize(self, target: TargetRegion) -> OcrResult:
        return OcrResult(
            region_id=target.region_id,
            backend=self.name,
            prompt_kind=_prompt_kind(target),
            text="",
            seconds=0.0,
            error="GLM-OCR backend disabled",
        )


class TransformersGlmOcrBackend(GlmOcrBackend):
    """CPU-local GLM-OCR backend using the official Transformers chat-template path."""

    name = "transformers"

    def __init__(self, settings: ParseSettings) -> None:
        self.settings = settings
        self._processor = None
        self._model = None
        self._torch = None

    def recognize(self, target: TargetRegion) -> OcrResult:
        started = time.perf_counter()
        prompt_kind = _prompt_kind(target)
        try:
            self._load()
            prompt = PROMPTS[prompt_kind]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": str(target.image_path)},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cpu")
            inputs.pop("token_type_ids", None)
            with self._torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.settings.glm_max_new_tokens,
                )
            input_len = inputs["input_ids"].shape[1]
            text = self._processor.decode(
                generated_ids[0][input_len:],
                skip_special_tokens=False,
            )
            return OcrResult(
                region_id=target.region_id,
                backend=self.name,
                prompt_kind=prompt_kind,
                text=text.strip(),
                seconds=time.perf_counter() - started,
            )
        except Exception as exc:  # noqa: BLE001 - convert backend failures to clear output.
            LOGGER.exception("GLM-OCR failed for %s", target.region_id)
            return OcrResult(
                region_id=target.region_id,
                backend=self.name,
                prompt_kind=prompt_kind,
                text="",
                seconds=time.perf_counter() - started,
                error=str(exc),
            )

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "The transformers GLM-OCR backend requires torch and transformers. "
                "Install requirements.txt first, or run with --glm-backend none for "
                "a Docling-only debug pass."
            ) from exc

        LOGGER.info("Loading GLM-OCR model %s on CPU", self.settings.glm_model)
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.settings.glm_model)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.settings.glm_model,
            torch_dtype=torch.float32,
            device_map=None,
        )
        self._model.to("cpu")
        self._model.eval()


class OllamaGlmOcrBackend(GlmOcrBackend):
    """Local Ollama backend for users who run the official glm-ocr Ollama model."""

    name = "ollama"

    def __init__(self, model: str = "glm-ocr", timeout_seconds: int = 1800) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds

    def recognize(self, target: TargetRegion) -> OcrResult:
        started = time.perf_counter()
        prompt_kind = _prompt_kind(target)
        prompt = f"{PROMPTS[prompt_kind]} {target.image_path}"
        try:
            completed = subprocess.run(
                ["ollama", "run", self.model, prompt],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            return OcrResult(
                region_id=target.region_id,
                backend=self.name,
                prompt_kind=prompt_kind,
                text="",
                seconds=time.perf_counter() - started,
                error="Ollama executable was not found on PATH.",
            )
        except Exception as exc:  # noqa: BLE001
            return OcrResult(
                region_id=target.region_id,
                backend=self.name,
                prompt_kind=prompt_kind,
                text="",
                seconds=time.perf_counter() - started,
                error=str(exc),
            )

        error = None
        if completed.returncode != 0:
            error = completed.stderr.strip() or f"ollama exited with {completed.returncode}"
        return OcrResult(
            region_id=target.region_id,
            backend=self.name,
            prompt_kind=prompt_kind,
            text=completed.stdout.strip(),
            seconds=time.perf_counter() - started,
            error=error,
        )


def build_glm_backend(settings: ParseSettings) -> GlmOcrBackend:
    backend = settings.glm_backend.lower()
    if backend == "none":
        return DisabledGlmOcrBackend()
    if backend == "transformers":
        return TransformersGlmOcrBackend(settings)
    if backend == "ollama":
        return OllamaGlmOcrBackend()
    raise ValueError(
        f"Unsupported GLM-OCR backend '{settings.glm_backend}'. "
        "Choose one of: transformers, ollama, none."
    )


def _prompt_kind(target: TargetRegion) -> str:
    if target.kind in PROMPTS:
        return target.kind
    if target.label == "formula":
        return "formula"
    if target.label == "table":
        return "table"
    return "text"

