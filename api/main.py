from __future__ import annotations

from fastapi import FastAPI

from api.dependencies import get_summarizer, is_model_loaded
from api.schemas import CompareModesRequest, CompareModesResponse, SummarizeRequest, SummarizeResponse
from smart_summarizer.constants import SUMMARY_MODES


app = FastAPI(
    title="Vietnamese Summarizer API",
    version="0.1.0",
    description="FastAPI backend for controllable Vietnamese abstractive summarization.",
)


@app.get("/api/health")
async def health() -> dict[str, bool | str]:
    return {"status": "ok", "model_loaded": is_model_loaded()}


@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest) -> SummarizeResponse:
    result = get_summarizer().generate_summary(
        text=request.text,
        mode=request.mode,
        length=request.length,
    )
    return SummarizeResponse(**result)


@app.post("/api/compare-modes", response_model=CompareModesResponse)
async def compare_modes(request: CompareModesRequest) -> CompareModesResponse:
    summarizer = get_summarizer()
    results = {
        mode: SummarizeResponse(
            **summarizer.generate_summary(
                text=request.text,
                mode=mode,
                length=request.length,
            )
        )
        for mode in SUMMARY_MODES
    }
    return CompareModesResponse(results=results)
