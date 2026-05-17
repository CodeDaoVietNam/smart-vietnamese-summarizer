from __future__ import annotations

import json
import socket
import sys
from pathlib import Path
from urllib import error, request

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.components import (
    LENGTH_HINTS,
    MODE_LABELS,
    MODE_HINTS,
    SAMPLES,
    SAMPLE_RECOMMENDATIONS,
    load_sample,
    render_info_card,
    render_keywords,
    render_quality_badge,
    render_summary_box,
)
from app.style import APP_CSS
from smart_summarizer.config import deep_get, load_config


APP_CONFIG = load_config("configs/app.yaml")
API_BASE_URL = deep_get(APP_CONFIG, "api.base_url", "http://127.0.0.1:8000")
API_TIMEOUT = float(deep_get(APP_CONFIG, "api.timeout_seconds", 180))


def run_generation(text: str, mode: str, length: str) -> dict:
    payload = json.dumps({"text": text, "mode": mode, "length": length}).encode("utf-8")
    http_request = request.Request(
        f"{API_BASE_URL}/api/summarize",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=API_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def run_mode_comparison(text: str, length: str) -> dict:
    payload = json.dumps({"text": text, "length": length}).encode("utf-8")
    http_request = request.Request(
        f"{API_BASE_URL}/api/compare-modes",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=API_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))["results"]


def render_api_error(exc: Exception) -> None:
    if isinstance(exc, error.HTTPError):
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        st.error(f"API returned HTTP {exc.code}. Detail: {detail}")
    elif isinstance(exc, (TimeoutError, socket.timeout)):
        st.error("Request timed out. The first model load can be slow; try again after the backend finishes loading.")
    else:
        st.error(
            f"Cannot reach backend at {API_BASE_URL}. "
            "Start it with `SUMMARIZER_CONFIG=configs/app_lora.yaml uvicorn api.main:app --reload`."
        )


def check_health() -> tuple[bool, str]:
    try:
        with request.urlopen(f"{API_BASE_URL}/api/health", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return bool(payload.get("model_loaded")), payload.get("status", "unknown")
    except (TimeoutError, socket.timeout, error.URLError):
        return False, "offline"


def main() -> None:
    st.set_page_config(
        page_title="Smart Vietnamese Summarizer",
        page_icon="📝",
        layout="wide",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Vietnamese Controllable Summarization</div>
            <div class="hero-title">Smart Summaries, Structured Outputs</div>
            <div class="hero-copy">
                Demo hệ thống tóm tắt tiếng Việt với ViT5/LoRA, FastAPI, Streamlit và structured formatter.
                Chọn một văn bản, đổi mode, rồi so sánh cách hệ thống sinh paragraph, bullet, action items và study notes.
            </div>
            <div class="pill-row">
                <span class="hero-pill">ViT5 backbone</span>
                <span class="hero-pill">LoRA adapter</span>
                <span class="hero-pill">Mode control</span>
                <span class="hero-pill">Formatter validation</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_loaded, status = check_health()
    if status == "ok" and model_loaded:
        st.success(f"Backend connected: {status}")
    elif status == "ok":
        st.info("Backend connected. Model will load on the first summarize request.")
    else:
        st.warning(
            f"Backend unavailable at {API_BASE_URL}. Start it with `uvicorn api.main:app --reload`."
        )

    with st.sidebar:
        st.subheader("Demo Playbook")
        render_info_card(
            "Best main demo",
            "Use Demo 1 for action items, Demo 3 for study notes, and Demo 7 for concise/bullet.",
        )
        render_info_card(
            "Honest limitation",
            "Heuristic Quality is not calibrated confidence. Outputs still need human review.",
        )
        st.caption(f"Backend URL: {API_BASE_URL}")

    sample_label = st.selectbox("Sample", list(SAMPLES.keys()), index=3)
    default_text = load_sample(sample_label)
    recommendation = SAMPLE_RECOMMENDATIONS.get(sample_label)
    if recommendation:
        st.info(recommendation)

    col_a, col_b, col_c = st.columns([1.2, 1, 1])
    with col_a:
        mode = st.radio(
            "Output mode",
            options=list(MODE_LABELS.keys()),
            format_func=lambda key: MODE_LABELS[key],
            index=0,
            horizontal=True,
        )
    with col_b:
        length = st.radio("Length", options=["short", "medium", "long"], index=1, horizontal=True)
    with col_c:
        compare = st.toggle("Compare all modes", value=False)

    hint_cols = st.columns(2)
    with hint_cols[0]:
        render_info_card("Mode behavior", MODE_HINTS[mode])
    with hint_cols[1]:
        render_info_card("Length behavior", LENGTH_HINTS[length])

    text = st.text_area(
        "Vietnamese input",
        value=default_text,
        height=260,
        placeholder="Dán meeting notes, lecture notes hoặc tài liệu tiếng Việt tại đây.",
    )
    word_count = len(text.split())
    st.caption(f"Input stats: {word_count} words, {len(text)} characters. Text is truncated to the model context window at inference.")

    summarize = st.button("Summarize", type="primary", use_container_width=True)

    if summarize:
        if not text.strip():
            st.warning("Vui lòng nhập văn bản tiếng Việt trước khi tóm tắt.")
            return

        left, right = st.columns(2)
        with left:
            st.subheader("Original")
            render_summary_box(text)

        with right:
            st.subheader("Generated")
            if compare:
                try:
                    with st.spinner("Generating all modes..."):
                        comparison = run_mode_comparison(text, length)
                except (TimeoutError, socket.timeout, error.URLError, error.HTTPError) as exc:
                    render_api_error(exc)
                    return
                tabs = st.tabs([MODE_LABELS[key] for key in MODE_LABELS])
                for tab, mode_key in zip(tabs, MODE_LABELS, strict=True):
                    with tab:
                        result = comparison[mode_key]
                        render_summary_box(result["summary"])
                        render_quality_badge(result["quality_estimate"])
                        metric_cols = st.columns(3)
                        metric_cols[0].metric("Input tokens", result["input_tokens"])
                        metric_cols[1].metric("Output tokens", result.get("output_tokens", 0))
                        metric_cols[2].metric("Latency", f"{result['latency_ms']} ms")
            else:
                try:
                    with st.spinner("Generating summary..."):
                        result = run_generation(text, mode or "concise", length or "medium")
                except (TimeoutError, socket.timeout, error.URLError, error.HTTPError) as exc:
                    render_api_error(exc)
                    return
                render_summary_box(result["summary"])

                metric_cols = st.columns(4)
                metric_cols[0].metric("Heuristic Quality", f"{result['quality_estimate']}%")
                metric_cols[1].metric("Input tokens", result["input_tokens"])
                metric_cols[2].metric("Output tokens", result.get("output_tokens", 0))
                metric_cols[3].metric("Latency", f"{result['latency_ms']} ms")
                render_quality_badge(result["quality_estimate"])

                st.subheader("Keywords")
                render_keywords(result["keywords"])
                st.markdown(
                    '<div class="small-note">Heuristic Quality is a debugging proxy based on length, repetition, keyword coverage, and generation signals. It is not a factuality score.</div>',
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
