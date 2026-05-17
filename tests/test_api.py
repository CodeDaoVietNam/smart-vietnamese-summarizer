from fastapi.testclient import TestClient

from api.main import app
from api.schemas import MAX_INPUT_CHARS


def test_health_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("api.main.get_summarizer", lambda: object())
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_summarize_endpoint(monkeypatch) -> None:
    class StubSummarizer:
        def generate_summary(self, text: str, mode: str, length: str) -> dict:
            return {
                "summary": "Tom tat",
                "keywords": ["tom", "tat"],
                "quality_estimate": 88.0,
                "latency_ms": 12,
                "input_tokens": 10,
                "mode": mode,
                "length": length,
            }

    monkeypatch.setattr("api.main.get_summarizer", lambda: StubSummarizer())
    client = TestClient(app)
    response = client.post(
        "/api/summarize",
        json={"text": "Noi dung", "mode": "concise", "length": "medium"},
    )
    assert response.status_code == 200
    assert response.json()["quality_estimate"] == 88.0


def test_compare_modes_endpoint(monkeypatch) -> None:
    class StubSummarizer:
        def generate_summary(self, text: str, mode: str, length: str) -> dict:
            return {
                "summary": f"Tom tat {mode}",
                "keywords": ["tom", "tat"],
                "quality_estimate": 80.0,
                "latency_ms": 10,
                "input_tokens": 8,
                "mode": mode,
                "length": length,
            }

    monkeypatch.setattr("api.main.get_summarizer", lambda: StubSummarizer())
    client = TestClient(app)
    response = client.post(
        "/api/compare-modes",
        json={"text": "Noi dung", "length": "medium"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert sorted(payload["results"]) == [
        "action_items",
        "bullet",
        "concise",
        "study_notes",
    ]
    assert payload["results"]["study_notes"]["summary"] == "Tom tat study_notes"


def test_summarize_rejects_overly_long_text(monkeypatch) -> None:
    monkeypatch.setattr("api.main.get_summarizer", lambda: object())
    client = TestClient(app)
    response = client.post(
        "/api/summarize",
        json={"text": "x" * (MAX_INPUT_CHARS + 1), "mode": "concise", "length": "medium"},
    )
    assert response.status_code == 422
