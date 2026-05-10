APP_CSS = """
<style>
    .stApp {
        background:
            linear-gradient(135deg, rgba(248, 250, 252, 0.95), rgba(240, 253, 250, 0.95)),
            radial-gradient(circle at 15% 10%, rgba(20, 184, 166, 0.10), transparent 30%),
            radial-gradient(circle at 85% 20%, rgba(245, 158, 11, 0.12), transparent 28%);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }
    .summary-box {
        border: 1px solid #d8e2dc;
        border-radius: 8px;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.82);
        min-height: 180px;
        white-space: pre-wrap;
    }
    .keyword-chip {
        display: inline-block;
        margin: 0.15rem 0.25rem 0.15rem 0;
        padding: 0.2rem 0.5rem;
        border-radius: 999px;
        background: #e0f2fe;
        color: #075985;
        font-size: 0.86rem;
    }
</style>
"""
