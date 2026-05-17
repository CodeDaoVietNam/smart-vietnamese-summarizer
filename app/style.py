APP_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@700;800&family=Nunito+Sans:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Nunito Sans', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 8% 12%, rgba(14, 165, 233, 0.16), transparent 26%),
            radial-gradient(circle at 92% 18%, rgba(245, 158, 11, 0.18), transparent 30%),
            linear-gradient(135deg, #f8fafc 0%, #eefdf8 46%, #fff7ed 100%);
    }
    h1, h2, h3 {
        font-family: 'Fraunces', serif;
        letter-spacing: -0.03em;
    }
    .hero-card {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 24px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1rem;
        color: #0f172a;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(240, 253, 250, 0.86)),
            radial-gradient(circle at 95% 0%, rgba(251, 191, 36, 0.35), transparent 32%);
        box-shadow: 0 22px 60px rgba(15, 23, 42, 0.10);
    }
    .hero-eyebrow {
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: #0f766e;
        margin-bottom: 0.35rem;
    }
    .hero-title {
        font-family: 'Fraunces', serif;
        font-size: clamp(2rem, 4vw, 3.7rem);
        line-height: 0.94;
        font-weight: 800;
        margin: 0 0 0.65rem 0;
    }
    .hero-copy {
        max-width: 760px;
        color: #334155;
        font-size: 1.02rem;
        line-height: 1.62;
    }
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.9rem;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        border: 1px solid rgba(13, 148, 136, 0.22);
        border-radius: 999px;
        padding: 0.28rem 0.7rem;
        background: rgba(240, 253, 250, 0.88);
        color: #0f766e;
        font-weight: 800;
        font-size: 0.82rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }
    .summary-box {
        border: 1px solid rgba(15, 23, 42, 0.10);
        border-radius: 18px;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.88);
        min-height: 180px;
        white-space: pre-wrap;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.07);
        line-height: 1.6;
    }
    .keyword-chip {
        display: inline-block;
        margin: 0.15rem 0.25rem 0.15rem 0;
        padding: 0.25rem 0.62rem;
        border-radius: 999px;
        background: #dff7ef;
        color: #0f766e;
        border: 1px solid rgba(13, 148, 136, 0.16);
        font-weight: 800;
        font-size: 0.86rem;
    }
    .info-card {
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 18px;
        padding: 0.85rem 1rem;
        margin: 0.6rem 0;
        background: rgba(255, 255, 255, 0.76);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    }
    .info-card-title {
        font-weight: 900;
        color: #0f172a;
        margin-bottom: 0.22rem;
    }
    .info-card-body {
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    .small-note {
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.45;
    }
</style>
"""
