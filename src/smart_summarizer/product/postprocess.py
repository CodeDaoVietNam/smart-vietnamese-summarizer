from __future__ import annotations

import re

from smart_summarizer.data.preprocessing import clean_text


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。])\s+")


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = SENTENCE_SPLIT_RE.split(text)
    if len(parts) == 1:
        parts = re.split(r"\s*[-•]\s*", text)
    return [part.strip(" -•\n\t") for part in parts if part.strip(" -•\n\t")]


def format_bullets(text: str) -> str:
    items = split_sentences(text)
    return "\n".join(f"- {item}" for item in items)


def format_action_items(text: str) -> str:
    items = split_sentences(text)
    if not items:
        return ""
    return "\n".join(f"- {item}" for item in items)


def format_study_notes(text: str) -> str:
    items = split_sentences(text)
    if not items:
        return ""
    return "\n".join(f"- {item}" for item in items)


def postprocess_summary(text: str, mode: str) -> str:
    text = clean_text(text)
    if mode == "bullet":
        return format_bullets(text)
    if mode == "action_items":
        return format_action_items(text)
    if mode == "study_notes":
        return format_study_notes(text)
    return text
