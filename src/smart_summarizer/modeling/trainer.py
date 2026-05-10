from __future__ import annotations

from typing import Any

from datasets import Dataset

from smart_summarizer.modeling.generation import build_instruction


def tokenize_seq2seq_dataset(
    dataset: Dataset,
    tokenizer,
    max_source_length: int,
    max_target_length: int,
    mode: str = "concise",
    prefixes: dict[str, str] | None = None,
) -> Dataset:
    def preprocess(batch: dict[str, list[str]]) -> dict[str, Any]:
        modes = batch.get("mode") or [mode] * len(batch["document"])
        inputs = [
            build_instruction(text, mode=sample_mode or mode, prefixes=prefixes)
            for text, sample_mode in zip(batch["document"], modes, strict=True)
        ]
        targets = batch["summary"]
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
