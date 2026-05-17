from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


MAX_INPUT_CHARS = 50_000


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    mode: Literal["concise", "bullet", "action_items", "study_notes"] = "concise"
    length: Literal["short", "medium", "long"] = "medium"


class CompareModesRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    length: Literal["short", "medium", "long"] = "medium"


class SummarizeResponse(BaseModel):
    summary: str
    keywords: list[str]
    quality_estimate: float
    latency_ms: int
    input_tokens: int
    output_tokens: int = 0
    mode: str
    length: str


class CompareModesResponse(BaseModel):
    results: dict[str, SummarizeResponse]
