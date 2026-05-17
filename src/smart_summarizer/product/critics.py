from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_summarizer.product.extractors import (
    ActionItem,
    STUDY_LABELS,
    StudyNotes,
    is_context_only_action,
    remove_instruction_leakage,
)


@dataclass(frozen=True)
class CriticReport:
    is_valid: bool
    warnings: list[str]
    repairs: list[str]


def critic_action_items(items: list[ActionItem], source: str = "") -> CriticReport:
    warnings: list[str] = []
    repairs: list[str] = []

    if not items:
        return CriticReport(is_valid=True, warnings=[], repairs=["no_clear_action_items"])

    for item in items:
        if item.owner == "Chưa rõ":
            warnings.append("missing_owner")
        if item.deadline == "Chưa rõ":
            warnings.append("missing_deadline")
        if is_context_only_action(item.evidence):
            warnings.append("context_only_item")
        if remove_instruction_leakage(item.action) != item.action:
            warnings.append("prompt_leakage")

    return CriticReport(is_valid=not any(warning != "missing_deadline" for warning in warnings), warnings=warnings, repairs=repairs)


def critic_study_notes(notes: StudyNotes) -> CriticReport:
    warnings: list[str] = []
    values = {
        "Khái niệm chính": notes.concept,
        "Cần nhớ": notes.remember,
        "Ví dụ": notes.example,
        "Lỗi dễ nhầm": notes.misconception,
    }

    for label, value in values.items():
        if not value or value == "Chưa nêu rõ trong văn bản":
            warnings.append(f"missing_study_field:{label}")
        if remove_instruction_leakage(value) != value:
            warnings.append("prompt_leakage")

    return CriticReport(is_valid=not warnings, warnings=warnings, repairs=[])


def validate_mode_output(output: str, mode: str) -> dict[str, Any]:
    lines = [line for line in output.splitlines() if line.strip()]
    missing: list[str] = []
    warnings: list[str] = []

    if mode == "concise":
        if output.strip().startswith("-"):
            warnings.append("concise_starts_with_bullet")
    elif mode == "bullet":
        bullet_lines = [line for line in lines if line.startswith("- ")]
        if len(bullet_lines) < 2:
            missing.append("bullet_lines")
    elif mode == "action_items":
        if output.strip() != "Không có việc cần làm rõ ràng.":
            for label in ("Người phụ trách:", "Hành động:", "Deadline:"):
                if label not in output:
                    missing.append(label.rstrip(":"))
    elif mode == "study_notes":
        for label in STUDY_LABELS:
            if f"{label}:" not in output:
                missing.append(label)

    return {
        "is_valid": not missing and not warnings,
        "missing_fields": missing,
        "warnings": warnings,
    }
