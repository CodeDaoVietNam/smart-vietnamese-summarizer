from __future__ import annotations

from pathlib import Path


def choose_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(model_name_or_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)


def load_seq2seq_model(model_name_or_path: str, device: str = "auto"):
    import torch
    from transformers import AutoModelForSeq2SeqLM

    resolved_device = choose_device(device)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model.to(torch.device(resolved_device))
    model.eval()
    return model, resolved_device


def resolve_existing_model(primary: str, fallback: str | None = None) -> str:
    if Path(primary).exists():
        return primary
    if fallback:
        return fallback
    return primary
