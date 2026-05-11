from __future__ import annotations

from typing import Any


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    if not predictions or not references:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    try:
        import evaluate

        if hasattr(evaluate, "load"):
            rouge = evaluate.load("rouge")
            scores: dict[str, Any] = rouge.compute(
                predictions=predictions,
                references=references,
                use_stemmer=False,
            )
            return {key: round(float(value), 4) for key, value in scores.items()}
    except Exception:
        pass

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=False,
    )
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    for prediction, reference in zip(predictions, references, strict=True):
        scores = scorer.score(reference, prediction)
        for key in totals:
            totals[key] += scores[key].fmeasure

    count = len(predictions)
    return {key: round(value / count, 4) for key, value in totals.items()}
