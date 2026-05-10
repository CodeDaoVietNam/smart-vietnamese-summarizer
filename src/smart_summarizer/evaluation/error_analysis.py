from __future__ import annotations

from collections import Counter


def repetition_ratio(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(tokens)


def length_ratio(source: str, summary: str) -> float:
    source_len = max(1, len(source.split()))
    return len(summary.split()) / source_len


def quick_error_tags(source: str, summary: str) -> list[str]:
    tags: list[str] = []
    ratio = length_ratio(source, summary)
    if ratio < 0.03:
        tags.append("too_short")
    if ratio > 0.7:
        tags.append("too_long")
    if repetition_ratio(summary) > 0.25:
        tags.append("repetition")
    if not summary.strip():
        tags.append("empty_output")
    return tags
