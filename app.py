# pyrefly: ignore [missing-import]
import streamlit as st

st.set_page_config(
    page_title="VisionTrace AI - Video Intelligence Platform",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

from frontend.dashboard import render_dashboard

if __name__ == "__main__":
    render_dashboard()
