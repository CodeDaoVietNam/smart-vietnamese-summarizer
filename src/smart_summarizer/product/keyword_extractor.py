from __future__ import annotations

import re
from collections import Counter

from smart_summarizer.constants import DEFAULT_STOPWORDS_VI
from smart_summarizer.data.preprocessing import clean_text


TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹ]+", re.UNICODE)


def extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    words = [word.lower() for word in TOKEN_RE.findall(clean_text(text))]
    candidates = [
        word
        for word in words
        if len(word) >= 3 and word not in DEFAULT_STOPWORDS_VI and not word.isdigit()
    ]
    counts = Counter(candidates)
    return [word for word, _ in counts.most_common(max_keywords)]
