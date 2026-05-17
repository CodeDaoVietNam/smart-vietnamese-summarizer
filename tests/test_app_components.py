from app.components import load_sample, render_summary_box


def test_load_sample_missing_label_returns_fallback() -> None:
    result = load_sample("Missing sample")
    assert "Không tìm thấy file mẫu" in result


def test_render_summary_box_preserves_newlines(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr("app.components.st.markdown", lambda value, **_: calls.append(value))
    render_summary_box("Dòng 1\nDòng 2")

    assert "<br>" in calls[0]
