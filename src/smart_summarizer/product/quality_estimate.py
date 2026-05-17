from __future__ import annotations

from smart_summarizer.evaluation.error_analysis import repetition_ratio


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def quality_estimate_from_scores(scores: list[float] | None) -> float | None:
    if not scores:
        return None
    # Generation scores from Hugging Face can be raw logits or log-probs depending on
    # decoding settings, so treat them only as a weak bounded confidence proxy.
    bounded = [clamp(score, -10.0, 0.0) for score in scores]
    mean_score = sum(bounded) / len(bounded)
    return round(((mean_score + 10.0) / 10.0) * 100.0, 2)


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
        scores.append(probability_score * 0.15)
    else:
        scores.append(10.0)

    ratio = len(summary.split()) / max(1, len(source.split()))
    scores.append(30.0 if 0.05 <= ratio <= 0.60 else 10.0)

    unique_part = clamp((1.0 - repetition_ratio(summary)) * 30.0, 0.0, 30.0)
    scores.append(unique_part)

    keyword_score = 15.0
    if keywords:
        summary_lower = summary.lower()
        covered = sum(1 for keyword in keywords if keyword.lower() in summary_lower)
        keyword_score = (covered / max(1, len(keywords))) * 25.0

    scores.append(keyword_score)
    return round(clamp(sum(scores)), 2)
