from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from smart_summarizer.data.preprocessing import clean_text, is_valid_pair
from smart_summarizer.utils.paths import ensure_parent, resolve_path


DOCUMENT_KEYS = ("document", "article", "content", "text", "body", "original", "source")
SUMMARY_KEYS = ("summary", "abstract", "highlights", "description", "target", "targets")
MODE_KEYS = ("mode", "task", "output_mode")


def _flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    return clean_text(str(value or ""))


def _first_present(example: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in example and example[key]:
            return _flatten_text(example[key])
    return ""


def normalize_pair(
    example: dict[str, Any],
    min_document_chars: int = 80,
    min_summary_chars: int = 20,
    max_document_chars: int | None = None,
) -> dict[str, str] | None:
    document = _first_present(example, DOCUMENT_KEYS)
    target = _first_present(example, SUMMARY_KEYS)
    if not is_valid_pair(
        document,
        target,
        min_document_chars=min_document_chars,
        min_summary_chars=min_summary_chars,
        max_document_chars=max_document_chars,
    ):
        return None
    row = {"document": document, "summary": target}
    mode = _first_present(example, MODE_KEYS)
    if mode:
        row["mode"] = mode
    return row


def load_wikilingua_vietnamese(language: str = "vietnamese") -> DatasetDict:
    """Load WikiLingua and normalize it into document/summary fields."""
    raw = load_dataset("wiki_lingua", language)
    normalized = {}
    for split_name, split in raw.items():
        rows = []
        for example in split:
            pair = normalize_pair(example)
            if pair:
                rows.append(pair)
        normalized[split_name] = Dataset.from_list(rows)
    return DatasetDict(normalized)


def load_vietnews(dataset_name: str) -> DatasetDict:
    raw = load_dataset(dataset_name)
    normalized = {}
    for split_name, split in raw.items():
        rows = []
        for example in split:
            pair = normalize_pair(example)
            if pair:
                rows.append(pair)
        normalized[split_name] = Dataset.from_list(rows)
    return DatasetDict(normalized)


def load_remote_dataset(dataset_name: str, language: str = "vietnamese") -> DatasetDict:
    if dataset_name in {"wiki_lingua", "esdurmus/wiki_lingua"}:
        return load_wikilingua_vietnamese(language)
    return load_vietnews(dataset_name)


def ensure_train_validation_test(dataset: DatasetDict, seed: int = 42) -> DatasetDict:
    if {"train", "validation", "test"}.issubset(dataset.keys()):
        return dataset

    if "train" not in dataset:
        first_split = next(iter(dataset.keys()))
        dataset = DatasetDict({"train": dataset[first_split]})

    first = dataset["train"].train_test_split(test_size=0.2, seed=seed)
    second = first["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict(
        {
            "train": first["train"],
            "validation": second["train"],
            "test": second["test"],
        }
    )


def save_jsonl(dataset: Dataset, output_path: str | Path) -> None:
    path = ensure_parent(output_path)
    with path.open("w", encoding="utf-8") as file:
        for row in dataset:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_jsonl_dataset(path: str | Path) -> Dataset:
    return Dataset.from_list(load_jsonl(path))
