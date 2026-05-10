from __future__ import annotations

from fastapi import FastAPI

from api.dependencies import get_summarizer
from api.schemas import SummarizeRequest, SummarizeResponse


app = FastAPI(
    title="Vietnamese Summarizer API",
    version="0.1.0",
    description="FastAPI backend for controllable Vietnamese abstractive summarization.",
)


@app.get("/api/health")
async def health() -> dict[str, bool | str]:
    model_loaded = get_summarizer() is not None
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest) -> SummarizeResponse:
    result = get_summarizer().generate_summary(
        text=request.text,
        mode=request.mode,
        length=request.length,
    )
    return SummarizeResponse(**result)
