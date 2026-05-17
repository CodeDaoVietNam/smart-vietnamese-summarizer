from __future__ import annotations

from dataclasses import dataclass

from smart_summarizer.constants import SUMMARY_LENGTHS, SUMMARY_MODES
from smart_summarizer.data.preprocessing import clean_text


MODE_PREFIXES = {
    "concise": "Tóm tắt ngắn gọn thành một đoạn văn tự nhiên, không dùng bullet",
    "bullet": "Tóm tắt thành các ý chính dạng bullet, mỗi bullet một ý",
    "action_items": "Chỉ trích xuất việc cần làm. Mỗi dòng gồm người phụ trách, hành động, deadline nếu có",
    "study_notes": "Tạo ghi chú học tập. Nêu khái niệm chính, cần nhớ, ví dụ, lỗi dễ nhầm",
}

DEFAULT_MAX_NEW_TOKENS = {
    "short": 64,
    "medium": 128,
    "long": 192,
}

DEFAULT_MIN_NEW_TOKENS = {
    "short": 0,
    "medium": 0,
    "long": 0,
}


@dataclass(frozen=True)
class GenerationRequest:
    text: str
    mode: str = "concise"
    length: str = "medium"


def validate_mode_length(mode: str, length: str) -> None:
    if mode not in SUMMARY_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Choose one of {SUMMARY_MODES}.")
    if length not in SUMMARY_LENGTHS:
        raise ValueError(f"Unsupported length '{length}'. Choose one of {SUMMARY_LENGTHS}.")


def build_instruction(
    text: str,
    mode: str = "concise",
    length: str = "medium",
    prefixes: dict[str, str] | None = None,
) -> str:
    validate_mode_length(mode, length)
    mode_prefixes = prefixes or MODE_PREFIXES
    prefix = mode_prefixes[mode].rstrip(": ")
    return f"{prefix}: {clean_text(text)}"


def generation_kwargs(
    length: str = "medium",
    num_beams: int = 4,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
    max_new_tokens: dict[str, int] | None = None,
    min_new_tokens: dict[str, int] | None = None,
    length_penalty: float = 1.0,
) -> dict[str, int | float | bool]:
    validate_mode_length("concise", length)
    max_token_map = max_new_tokens or DEFAULT_MAX_NEW_TOKENS
    min_token_map = min_new_tokens or DEFAULT_MIN_NEW_TOKENS
    return {
        "max_new_tokens": int(max_token_map.get(length, DEFAULT_MAX_NEW_TOKENS[length])),
        "min_new_tokens": int(min_token_map.get(length, DEFAULT_MIN_NEW_TOKENS[length])),
        "num_beams": num_beams,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "length_penalty": length_penalty,
        "early_stopping": True,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
