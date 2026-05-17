from __future__ import annotations

import html
from pathlib import Path

import streamlit as st


SAMPLES = {
    "Meeting note": "data/samples/meeting_note_vi.txt",
    "Lecture note": "data/samples/lecture_note_vi.txt",
    "Article": "data/samples/article_vi.txt",
    "Demo 1 - Meeting release": "data/samples/web_demo/01_meeting_demo_release.txt",
    "Demo 2 - Meeting without clear tasks": "data/samples/web_demo/02_meeting_no_clear_tasks.txt",
    "Demo 3 - Transformer lecture": "data/samples/web_demo/03_lecture_transformer_attention.txt",
    "Demo 4 - ML evaluation lecture": "data/samples/web_demo/04_lecture_ml_evaluation.txt",
    "Demo 5 - Project update": "data/samples/web_demo/05_project_update_summarizer.txt",
    "Demo 6 - Project risk status": "data/samples/web_demo/06_project_risk_status.txt",
    "Demo 7 - Urban transport article": "data/samples/web_demo/07_article_urban_transport.txt",
    "Demo 8 - OS cache study material": "data/samples/web_demo/08_study_material_os_cache.txt",
}

MODE_LABELS = {
    "concise": "Concise paragraph",
    "bullet": "Bullet summary",
    "action_items": "Action items",
    "study_notes": "Study notes",
}

MODE_HINTS = {
    "concise": "Best for articles or general documents where you need a compact paragraph.",
    "bullet": "Best for project updates, articles, or lectures where key points matter.",
    "action_items": "Best for meetings with owners, tasks, and deadlines.",
    "study_notes": "Best for lectures or study materials with concepts, examples, and common mistakes.",
}

LENGTH_HINTS = {
    "short": "Shortest useful output. Good for quick preview.",
    "medium": "Recommended for demo. Balanced detail and readability.",
    "long": "More complete output. Useful when the source has several distinct points.",
}

SAMPLE_RECOMMENDATIONS = {
    "Demo 1 - Meeting release": "Recommended: Action items, then Compare all modes.",
    "Demo 3 - Transformer lecture": "Recommended: Study notes or Bullet summary.",
    "Demo 5 - Project update": "Recommended: Bullet summary; use Action items as a limitation case.",
    "Demo 7 - Urban transport article": "Recommended: Concise paragraph or Bullet summary.",
    "Demo 8 - OS cache study material": "Recommended: Study notes.",
}


def load_sample(label: str) -> str:
    path = Path(SAMPLES.get(label, ""))
    if not path.is_file():
        return f"[Không tìm thấy file mẫu '{label}'. Bạn vẫn có thể dán văn bản thủ công tại đây.]"
    return path.read_text(encoding="utf-8")


def render_summary_box(text: str) -> None:
    escaped = html.escape(text or "Chưa có kết quả.").replace("\n", "<br>")
    st.markdown(f'<div class="summary-box">{escaped}</div>', unsafe_allow_html=True)


def render_keywords(keywords: list[str]) -> None:
    if not keywords:
        st.caption("No keywords detected yet.")
        return
    chips = "".join(
        f'<span class="keyword-chip">{html.escape(keyword)}</span>' for keyword in keywords
    )
    st.markdown(chips, unsafe_allow_html=True)


def render_quality_badge(score: float) -> None:
    st.progress(
        min(max(score / 100.0, 0.0), 1.0),
        text=f"Heuristic Quality Estimate: {score}%",
    )


def render_info_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-title">{html.escape(title)}</div>
            <div class="info-card-body">{html.escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
