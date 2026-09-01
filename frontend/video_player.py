import streamlit as st
from models.schemas import VideoMetadata
from utils.time_utils import seconds_to_timestamp
from frontend.components import render_metric_card

def render_video_player(metadata: VideoMetadata) -> None:
    """Render video player and metadata dashboard."""
    col1, col2 = st.columns([7, 5])

    with col1:
        st.subheader("🎥 Video Player")
        st.markdown('<div class="video-wrapper">', unsafe_allow_html=True)
        st.video(metadata.filepath)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("📊 Video Information")
        st.markdown(f"**Filename:** `{metadata.filename}`")
        st.markdown(f"**Video Hash:** `{metadata.video_hash[:12]}...`")

        mcol1, mcol2 = st.columns(2)
        with mcol1:
            render_metric_card("Duration", seconds_to_timestamp(metadata.duration_sec), f"{metadata.duration_sec:.1f} seconds")
            render_metric_card("FPS", f"{metadata.fps:.2f}", "Frames Per Second")
        with mcol2:
            render_metric_card("Resolution", metadata.resolution_str, f"{metadata.width}x{metadata.height}")
            render_metric_card("Frame Count", f"{metadata.frame_count:,}", f"Codec: {metadata.codec}")

        st.markdown(f"**File Size:** `{metadata.file_size_mb:.2f} MB`")
