from __future__ import annotations

from typing import Any

from smart_summarizer.data.preprocessing import clean_text
from smart_summarizer.product.critics import (
    critic_action_items,
    critic_study_notes,
    validate_mode_output as critic_validate_mode_output,
)
from smart_summarizer.product.extractors import (
    dedupe_near_preserve_order,
    dedupe_preserve_order,
    extract_action_items,
    extract_study_notes,
    first_sentences,
    split_sentences,
    strip_mode_labels,
)
from smart_summarizer.product.renderers import (
    render_action_items,
    render_bullets,
    render_concise,
    render_study_notes,
)


def format_concise(text: str, source: str | None = None, length: str = "medium") -> str:
    items = [strip_mode_labels(item) for item in split_sentences(text)]
    items = [item for item in dedupe_preserve_order(items) if item]
    if source:
        items.extend(first_sentences(source, limit=4))
        items = dedupe_preserve_order([strip_mode_labels(item) for item in items])
    return render_concise(items, length=length)


def format_bullet(text: str, source: str | None = None, length: str = "medium") -> str:
    items = [strip_mode_labels(item) for item in split_sentences(text)]
    items = [item for item in dedupe_preserve_order(items) if item]
    if len(items) < 2 and source:
        items.extend(first_sentences(source, limit=6))
        items = dedupe_preserve_order([strip_mode_labels(item) for item in items])
    items = dedupe_near_preserve_order(items)
    return render_bullets(items, length=length)


def format_action_items(text: str, source: str | None = None, length: str = "medium") -> str:
    items = extract_action_items(source=source or "", draft=text, length=length)
    critic_action_items(items, source=source or "")
    return render_action_items(items)


def format_study_notes(text: str, source: str | None = None, length: str = "medium") -> str:
    notes = extract_study_notes(source=source or "", draft=text, length=length)
    critic_study_notes(notes)
    return render_study_notes(notes)


def validate_mode_output(output: str, mode: str) -> dict[str, Any]:
    return critic_validate_mode_output(output, mode)


def format_summary(text: str, source: str, mode: str, length: str = "medium") -> str:
    if mode == "concise":
        output = format_concise(text, source=source, length=length)
    elif mode == "bullet":
        output = format_bullet(text, source=source, length=length)
    elif mode == "action_items":
        output = format_action_items(text, source=source, length=length)
    elif mode == "study_notes":
        output = format_study_notes(text, source=source, length=length)
    else:
        output = clean_text(text)

    validation = validate_mode_output(output, mode)
    if validation["is_valid"]:
        return output

    # One repair pass using the source as a fallback for weak generations.
    if mode == "bullet":
        return format_bullet(f"{output}\n{source}", source=source, length=length)
    if mode == "action_items":
        return format_action_items(f"{output}\n{source}", source=source, length=length)
    if mode == "study_notes":
        return format_study_notes(f"{output}\n{source}", source=source, length=length)
    if mode == "concise":
        return format_concise(output, source=source, length=length)
    return output
