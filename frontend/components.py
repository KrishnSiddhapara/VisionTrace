import streamlit as st

def load_custom_css() -> None:
    """Inject custom modern dark theme styling into Streamlit."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        /* Root color palette */
        :root {
            --bg-dark: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.7);
            --bg-card-hover: rgba(51, 65, 85, 0.8);
            --border-glass: rgba(255, 255, 255, 0.1);
            --primary-accent: #6366f1;
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --font-family: 'Plus Jakarta Sans', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        /* Global App Styling */
        .stApp {
            font-family: var(--font-family);
            background-color: var(--bg-dark);
            color: var(--text-main);
        }

        /* Main Container padding */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1300px;
        }

        /* Header Hero Banner */
        .hero-container {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid var(--border-glass);
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin-bottom: 2rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            position: relative;
            overflow: hidden;
        }

        .hero-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--primary-gradient);
        }

        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .hero-subtitle {
            color: var(--text-muted);
            font-size: 1.05rem;
            margin-top: 0.5rem;
            font-weight: 400;
        }

        /* Metric Cards */
        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-glass);
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            backdrop-filter: blur(8px);
            transition: all 0.25s ease-in-out;
        }

        .metric-card:hover {
            background: var(--bg-card-hover);
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.4);
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.15);
        }

        .metric-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            font-weight: 600;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--text-main);
            font-family: var(--font-mono);
        }

        .metric-sub {
            font-size: 0.8rem;
            color: var(--accent-cyan);
            margin-top: 0.2rem;
        }

        /* Glassmorphism Containers */
        .glass-box {
            background: var(--bg-card);
            border: 1px solid var(--border-glass);
            border-radius: 14px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            margin-bottom: 1.5rem;
        }

        /* Badges */
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            gap: 6px;
        }

        .badge-success {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .badge-info {
            background: rgba(99, 102, 241, 0.15);
            color: #818cf8;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }

        .badge-warning {
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        /* Custom Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(15, 23, 42, 0.6);
            padding: 8px;
            border-radius: 12px;
            border: 1px solid var(--border-glass);
        }

        .stTabs [data-baseweb="tab"] {
            height: 42px;
            white-space: pre;
            border-radius: 8px;
            color: var(--text-muted);
            font-weight: 500;
            padding: 0 16px;
        }

        .stTabs [aria-selected="true"] {
            background: var(--primary-gradient) !important;
            color: #ffffff !important;
            font-weight: 700;
        }

        /* Streamlit Uploader Styling */
        section[data-testid="stFileUploader"] {
            background: rgba(30, 41, 59, 0.5);
            border: 2px dashed rgba(99, 102, 241, 0.4);
            border-radius: 16px;
            padding: 1.5rem;
            transition: border-color 0.3s ease;
        }

        section[data-testid="stFileUploader"]:hover {
            border-color: var(--primary-accent);
        }

        /* Video Container */
        .video-wrapper {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid var(--border-glass);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_header() -> None:
    """Render application hero banner."""
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">
                🎥 VisionTrace AI
                <span class="status-badge badge-info">VLM Video Intelligence</span>
            </div>
            <div class="hero-subtitle">
                Advanced Visual Understanding, Object Tracking, Event Reasoning & Natural Language Video Q&A Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric_card(label: str, value: str, subtext: str = "") -> None:
    """Render a styled metric card."""
    sub_html = f'<div class="metric-sub">{subtext}</div>' if subtext else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_status_badge(text: str, badge_type: str = "info") -> str:
    """Return HTML string for status badge."""
    return f'<span class="status-badge badge-{badge_type}">{text}</span>'
