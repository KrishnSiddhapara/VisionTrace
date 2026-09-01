import streamlit as st
from typing import List, Dict, Any

def render_timeline_ui(timeline: List[Dict[str, Any]]) -> None:
    """Render interactive chronological timeline UI."""
    st.subheader("⏳ Chronological Video Timeline")

    if not timeline:
        st.info("No timeline events generated yet.")
        return

    for step in timeline:
        t_str = step.get("formatted_time", "00:00")
        evt_type = step.get("event_type", "INFO")
        desc = step.get("description", "")
        subject = step.get("subject", "")

        conf = step.get("confidence", 0.9)

        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.6); border-left: 4px solid #6366f1; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #818cf8; font-size: 1.1rem;">⏱️ {t_str}</span>
                    <span style="background: rgba(99, 102, 241, 0.2); color: #c7d2fe; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">{evt_type}</span>
                </div>
                <div style="margin-top: 6px; font-size: 1rem; color: #f8fafc; font-weight: 500;">{desc}</div>
                <div style="margin-top: 4px; font-size: 0.8rem; color: #94a3b8;">Subject: <b>{subject or 'N/A'}</b> | Confidence: <b>{int(conf*100)}%</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )
