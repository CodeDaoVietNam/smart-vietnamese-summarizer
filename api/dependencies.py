from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smart_summarizer.product.summarizer import SmartSummarizer


@lru_cache(maxsize=1)
def get_summarizer() -> SmartSummarizer:
    return SmartSummarizer.from_config("configs/app.yaml")
