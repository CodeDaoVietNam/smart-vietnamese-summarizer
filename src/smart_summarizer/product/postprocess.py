from __future__ import annotations

import re

from smart_summarizer.data.preprocessing import clean_text


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。])\s+")


def split_sentences(text: str) -> list[str]:
    line_items = [
        clean_text(line).strip(" -•\n\t")
        for line in (text or "").splitlines()
        if clean_text(line).strip(" -•\n\t")
    ]
    if len(line_items) > 1:
        return line_items

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
    if "không có việc cần làm" in text.lower():
        return "Không có việc cần làm rõ ràng."
    items = split_sentences(text)
    if not items:
        return ""
    return "\n".join(f"- {item}" for item in items)


def format_study_notes(text: str) -> str:
    items = split_sentences(text)
    if not items:
        return ""

    label_prefixes = (
        "khái niệm chính:",
        "cần nhớ:",
        "ví dụ:",
        "ý nghĩa:",
        "lỗi dễ nhầm:",
    )
    if any(item.lower().startswith(label_prefixes) for item in items):
        return "\n".join(items)

    labels = ("Khái niệm chính", "Cần nhớ", "Ví dụ", "Lỗi dễ nhầm")
    return "\n".join(
        f"{label}: {item}"
        for label, item in zip(labels, items[: len(labels)], strict=False)
    )


def postprocess_summary(text: str, mode: str) -> str:
    if mode == "bullet":
        return format_bullets(text)
    if mode == "action_items":
        return format_action_items(text)
    if mode == "study_notes":
        return format_study_notes(text)
    return clean_text(text)
