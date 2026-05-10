from __future__ import annotations

import re
import unicodedata


WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip()


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", normalize_whitespace(text))
    text = text.replace(" ,", ",").replace(" .", ".")
    return text


def is_valid_pair(
    document: str,
    summary: str,
    min_document_chars: int = 80,
    min_summary_chars: int = 20,
    max_document_chars: int | None = None,
) -> bool:
    document = clean_text(document)
    summary = clean_text(summary)
    if len(document) < min_document_chars or len(summary) < min_summary_chars:
        return False
    if max_document_chars is not None and len(document) > max_document_chars:
        return False
    return True


def truncate_by_words(text: str, max_words: int) -> str:
    words = clean_text(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])
