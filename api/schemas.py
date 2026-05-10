from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1)
    mode: Literal["concise", "bullet", "action_items", "study_notes"] = "concise"
    length: Literal["short", "medium", "long"] = "medium"


class SummarizeResponse(BaseModel):
    summary: str
    keywords: list[str]
    quality_estimate: float
    latency_ms: int
    input_tokens: int
    mode: str
    length: str
