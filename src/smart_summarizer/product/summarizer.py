from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from smart_summarizer.config import deep_get, load_config
from smart_summarizer.data.preprocessing import clean_text
from smart_summarizer.modeling.generation import build_instruction, generation_kwargs, validate_mode_length
from smart_summarizer.modeling.model_loader import load_seq2seq_model, load_tokenizer, resolve_existing_model
from smart_summarizer.product.keyword_extractor import extract_keywords
from smart_summarizer.product.postprocess import postprocess_summary
from smart_summarizer.product.quality_estimate import compute_quality_estimate


class SmartSummarizer:
    def __init__(
        self,
        model_name_or_path: str,
        fallback_model: str | None = None,
        device: str = "auto",
        max_source_length: int = 512,
        generation_config: dict[str, Any] | None = None,
    ) -> None:
        self.model_name_or_path = resolve_existing_model(model_name_or_path, fallback_model)
        self.device = device
        self.max_source_length = max_source_length
        self.generation_config = generation_config or {}
        self.prefixes = self.generation_config.get("prefixes")
        self.tokenizer = load_tokenizer(self.model_name_or_path)
        self.model, self.resolved_device = load_seq2seq_model(self.model_name_or_path, device=device)

    @classmethod
    def from_config(cls, config_path: str | Path = "configs/app.yaml") -> "SmartSummarizer":
        config = load_config(config_path)
        return cls(
            model_name_or_path=deep_get(config, "model.name", "models/vit5-summarizer-v2"),
            fallback_model=deep_get(config, "model.fallback_name", "VietAI/vit5-base"),
            device=deep_get(config, "model.device", "auto"),
            max_source_length=int(deep_get(config, "tokenization.max_source_length", 512)),
            generation_config=deep_get(config, "generation", {}),
        )

    def generate_summary(self, text: str, mode: str = "concise", length: str = "medium") -> dict[str, Any]:
        validate_mode_length(mode, length)
        cleaned = clean_text(text)
        if not cleaned:
            return {
                "summary": "",
                "keywords": [],
                "quality_estimate": 0.0,
                "latency_ms": 0,
                "input_tokens": 0,
                "mode": mode,
                "length": length,
            }

        instruction = build_instruction(cleaned, mode=mode, prefixes=self.prefixes)
        encoded = self.tokenizer(
            instruction,
            return_tensors="pt",
            max_length=self.max_source_length,
            truncation=True,
        )
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}

        kwargs = generation_kwargs(
            length=length,
            num_beams=int(self.generation_config.get("num_beams", 4)),
            repetition_penalty=float(self.generation_config.get("repetition_penalty", 1.2)),
            no_repeat_ngram_size=int(self.generation_config.get("no_repeat_ngram_size", 3)),
            max_new_tokens=self.generation_config.get("max_new_tokens"),
        )

        start = time.perf_counter()
        outputs = self.model.generate(**encoded, **kwargs)
        latency_ms = int((time.perf_counter() - start) * 1000)

        generated = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        summary = postprocess_summary(generated, mode)
        keywords = extract_keywords(cleaned)
        score_tensors = getattr(outputs, "scores", None) or []
        generation_scores = [float(score.max().item()) for score in score_tensors]
        quality_estimate = compute_quality_estimate(
            cleaned,
            summary,
            keywords,
            generation_scores=generation_scores,
        )

        return {
            "summary": summary,
            "keywords": keywords,
            "quality_estimate": quality_estimate,
            "latency_ms": latency_ms,
            "input_tokens": int(encoded["input_ids"].shape[-1]),
            "mode": mode,
            "length": length,
        }


_DEFAULT_SUMMARIZER: SmartSummarizer | None = None


def get_default_summarizer(config_path: str | Path = "configs/app.yaml") -> SmartSummarizer:
    global _DEFAULT_SUMMARIZER
    if _DEFAULT_SUMMARIZER is None:
        _DEFAULT_SUMMARIZER = SmartSummarizer.from_config(config_path)
    return _DEFAULT_SUMMARIZER


def generate_summary(text: str, mode: str = "concise", length: str = "medium") -> dict[str, Any]:
    return get_default_summarizer().generate_summary(text, mode=mode, length=length)
