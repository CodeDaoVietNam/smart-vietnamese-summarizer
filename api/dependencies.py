from __future__ import annotations

from functools import lru_cache

from smart_summarizer.product.summarizer import SmartSummarizer


@lru_cache(maxsize=1)
def get_summarizer() -> SmartSummarizer:
    return SmartSummarizer.from_config("configs/app.yaml")
