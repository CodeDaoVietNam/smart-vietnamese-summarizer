from __future__ import annotations

from smart_summarizer.data.preprocessing import clean_text
from smart_summarizer.product.extractors import (
    ActionItem,
    LENGTH_ITEM_LIMITS,
    LENGTH_SENTENCE_LIMITS,
    StudyNotes,
    length_value,
)


def render_action_items(items: list[ActionItem]) -> str:
    if not items:
        return "Không có việc cần làm rõ ràng."
    return "\n".join(
        (
            f"- Người phụ trách: {item.owner}\n"
            f"  Hành động: {item.action}\n"
            f"  Deadline: {item.deadline}"
        )
        for item in items
    )


def render_study_notes(notes: StudyNotes) -> str:
    return "\n".join(
        (
            f"Khái niệm chính: {notes.concept}",
            f"Cần nhớ: {notes.remember}",
            f"Ví dụ: {notes.example}",
            f"Lỗi dễ nhầm: {notes.misconception}",
        )
    )


def render_bullets(items: list[str], length: str = "medium") -> str:
    limit = length_value(length, LENGTH_ITEM_LIMITS)
    return "\n".join(f"- {clean_text(item)}" for item in items[:limit])


def render_concise(items: list[str], length: str = "medium") -> str:
    limit = length_value(length, LENGTH_SENTENCE_LIMITS)
    return clean_text(" ".join(items[:limit]))
