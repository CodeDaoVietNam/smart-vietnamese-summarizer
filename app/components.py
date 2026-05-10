from __future__ import annotations

import html
from pathlib import Path

import streamlit as st


SAMPLES = {
    "Meeting note": "data/samples/meeting_note_vi.txt",
    "Lecture note": "data/samples/lecture_note_vi.txt",
    "Article": "data/samples/article_vi.txt",
}

MODE_LABELS = {
    "concise": "Concise",
    "bullet": "Bullet",
    "action_items": "Action items",
    "study_notes": "Study notes",
}


def load_sample(label: str) -> str:
    path = Path(SAMPLES[label])
    return path.read_text(encoding="utf-8")


def render_summary_box(text: str) -> None:
    escaped = html.escape(text or "Chưa có kết quả.")
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
    st.progress(min(max(score / 100.0, 0.0), 1.0), text=f"Quality Estimate: {score}%")
