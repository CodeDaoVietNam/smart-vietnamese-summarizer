from __future__ import annotations

import math

from smart_summarizer.evaluation.error_analysis import repetition_ratio


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def quality_estimate_from_scores(scores: list[float] | None) -> float | None:
    if not scores:
        return None
    avg = sum(scores) / len(scores)
    return round(clamp(math.exp(avg) * 100), 2)


def compute_quality_estimate(
    source: str,
    summary: str,
    keywords: list[str] | None = None,
    generation_scores: list[float] | None = None,
) -> float:
    if not source.strip() or not summary.strip():
        return 0.0

    scores: list[float] = []
    probability_score = quality_estimate_from_scores(generation_scores)
    if probability_score is not None:
        scores.append(probability_score * 0.30)
    else:
        scores.append(22.0)

    ratio = len(summary.split()) / max(1, len(source.split()))
    scores.append(25.0 if 0.05 <= ratio <= 0.50 else 8.0)

    unique_part = clamp((1.0 - repetition_ratio(summary)) * 25.0, 0.0, 25.0)
    scores.append(unique_part)

    keyword_score = 12.0
    if keywords:
        summary_lower = summary.lower()
        covered = sum(1 for keyword in keywords if keyword.lower() in summary_lower)
        keyword_score = (covered / max(1, len(keywords))) * 20.0

    scores.append(keyword_score)
    return round(clamp(sum(scores)), 2)
