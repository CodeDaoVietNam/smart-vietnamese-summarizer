from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib import error, request

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.components import (
    MODE_LABELS,
    SAMPLES,
    load_sample,
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


def check_health() -> tuple[bool, str]:
    try:
        with request.urlopen(f"{API_BASE_URL}/api/health", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return bool(payload.get("model_loaded")), payload.get("status", "unknown")
    except error.URLError:
        return False, "offline"


def main() -> None:
    st.set_page_config(
        page_title="Smart Vietnamese Summarizer",
        page_icon="📝",
        layout="wide",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.title("Smart Meeting & Study Notes Summarization")
    st.caption("Vietnamese controllable summarization powered by a fine-tuned Transformer model.")

    model_loaded, status = check_health()
    if model_loaded:
        st.success(f"Backend connected: {status}")
    else:
        st.warning(
            f"Backend unavailable at {API_BASE_URL}. Start it with `uvicorn api.main:app --reload`."
        )

    sample_label = st.selectbox("Sample", list(SAMPLES.keys()), index=0)
    default_text = load_sample(sample_label)

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

    text = st.text_area(
        "Vietnamese input",
        value=default_text,
        height=260,
        placeholder="Dán meeting notes, lecture notes hoặc tài liệu tiếng Việt tại đây.",
    )

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
                tabs = st.tabs([MODE_LABELS[key] for key in MODE_LABELS])
                for tab, mode_key in zip(tabs, MODE_LABELS, strict=True):
                    with tab:
                        with st.spinner(f"Generating {MODE_LABELS[mode_key]}..."):
                            result = run_generation(text, mode_key, length)
                        render_summary_box(result["summary"])
                        render_quality_badge(result["quality_estimate"])
            else:
                with st.spinner("Generating summary..."):
                    result = run_generation(text, mode or "concise", length or "medium")
                render_summary_box(result["summary"])

                metric_cols = st.columns(3)
                metric_cols[0].metric("Quality Estimate", f"{result['quality_estimate']}%")
                metric_cols[1].metric("Input tokens", result["input_tokens"])
                metric_cols[2].metric("Latency", f"{result['latency_ms']} ms")
                render_quality_badge(result["quality_estimate"])

                st.subheader("Keywords")
                render_keywords(result["keywords"])


if __name__ == "__main__":
    main()
