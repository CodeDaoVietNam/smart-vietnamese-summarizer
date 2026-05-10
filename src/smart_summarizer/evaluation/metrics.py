from __future__ import annotations

from typing import Any


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    import evaluate

    rouge = evaluate.load("rouge")
    scores: dict[str, Any] = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=False,
    )
    return {key: round(float(value), 4) for key, value in scores.items()}
