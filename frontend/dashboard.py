from pathlib import Path
import streamlit as st

from config.settings import settings
from video.processor import video_processor
from video.sampler import frame_sampler
from vision.frame_analyzer import frame_analyzer
from vision.object_detector import object_detector
from vision.tracker import object_tracker
from intelligence.event_detector import event_detector
from intelligence.temporal_reasoner import temporal_reasoner
from intelligence.key_moment_detector import key_moment_detector
from intelligence.anomaly_detector import anomaly_detector
from intelligence.video_memory import video_memory_manager
from models.schemas import VideoMemory
from qa.video_qa import video_qa_engine
from retrieval.semantic_search import semantic_search_engine
from reports.report_generator import report_generator

from frontend.components import load_custom_css, render_header, render_metric_card
from frontend.video_player import render_video_player
from frontend.timeline_ui import render_timeline_ui
from frontend.analytics_ui import (
    render_objects_and_tracks_ui,
    render_scenes_ui,
    render_quality_control_panel,
    render_bbox_debug_visualizer,
)
from utils.logger import logger

def init_session_state() -> None:
    """Initialize Streamlit session state keys."""
    if "metadata" not in st.session_state:
        st.session_state["metadata"] = None
    if "video_path" not in st.session_state:
        st.session_state["video_path"] = None
    if "memory" not in st.session_state:
        st.session_state["memory"] = None
    if "sampling_mode" not in st.session_state:
        st.session_state["sampling_mode"] = "Balanced"


def run_full_pipeline(video_path: Path, sampling_mode: str = "Balanced") -> VideoMemory:
    """Execute complete 12-phase accuracy upgrade processing pipeline."""
    progress_bar = st.progress(0, text="Initializing high-accuracy video processing pipeline...")

    # Step 1: Validation & Metadata
    progress_bar.progress(10, text="✓ Step 1/7: Validating video & extracting metadata...")
    validation, metadata, scenes, _ = video_processor.process_video(video_path)

    # Check cached memory
    cache_key_hash = f"{metadata.video_hash}_{sampling_mode.lower()}"
    existing_mem = video_memory_manager.load_memory(cache_key_hash)
    if existing_mem:
        progress_bar.progress(100, text="✓ Loaded cached VideoMemory.")
        return existing_mem

    # Step 2: Intelligent Multi-Criteria Sampler
    progress_bar.progress(25, text=f"● Step 2/7: Extracting intelligent representative frames ({sampling_mode} Mode)...")
    output_frames_dir = settings.PROCESSED_DIR / metadata.video_hash / "frames"
    sampled_frames = frame_sampler.sample_scene_frames(video_path, scenes, output_frames_dir, sampling_mode=sampling_mode)

    # Step 3: Multi-Frame Window VLM Frame Analysis
    progress_bar.progress(45, text="● Step 3/7: Analyzing frames with Vision-Language Model & window context...")
    frame_obs_list = []
    prev_obs = None
    for sf in sampled_frames:
        obs = frame_analyzer.analyze_frame(sf, prev_obs=prev_obs)
        frame_obs_list.append(obs)
        prev_obs = obs

    # Step 4: YOLO Object Detection
    progress_bar.progress(60, text="● Step 4/7: Running YOLO object detection...")
    yolo_dets = {}
    for sf in sampled_frames:
        dets = object_detector.detect_objects(sf.path)
        yolo_dets[sf.frame_id] = dets

    # Step 5: Bounding Box IoU Spatial Tracking
    progress_bar.progress(75, text="● Step 5/7: Tracking objects with BBox IoU spatial matcher...")
    tracks = object_tracker.track_entities(sampled_frames, yolo_dets, frame_obs_list)

    # Step 6: Multi-Frame Temporal Event Verification
    progress_bar.progress(85, text="● Step 6/7: Verifying physical state changes & timed events...")
    events = event_detector.detect_events(scenes, frame_obs_list, tracks)

    # Step 7: Temporal Reasoning & Memory
    progress_bar.progress(95, text="● Step 7/7: Synthesizing timeline & structured memory...")
    timeline = temporal_reasoner.synthesize_timeline(scenes, events, tracks, frame_obs_list)
    summaries = temporal_reasoner.generate_summaries(metadata, scenes, timeline, tracks)
    key_moments = key_moment_detector.detect_key_moments(events, tracks, scenes)
    anomalies = anomaly_detector.detect_anomalies(frame_obs_list, tracks)

    memory = VideoMemory(
        video_hash=cache_key_hash,
        metadata=metadata,
        scenes=scenes,
        sampled_frames=sampled_frames,
        frame_observations=frame_obs_list,
        yolo_detections=yolo_dets,
        tracks=tracks,
        events=events,
        timeline=timeline,
        summary=summaries,
        insights=[],
    )

    video_memory_manager.save_memory(memory)
    progress_bar.progress(100, text="✅ High-Accuracy Pipeline Complete!")
    return memory


def render_dashboard() -> None:
    """Main dashboard rendering function."""
    load_custom_css()
    render_header()
    init_session_state()

    # Sidebar settings & info
    with st.sidebar:
        st.title("⚙️ Platform Config")
        sampling_mode = st.radio(
            "🎯 Sampling Profile Profile:",
            ["Balanced", "Fast", "Deep Analysis"],
            index=0,
            help="Balanced: Optimal accuracy/speed. Fast: Quick overview. Deep Analysis: Dense sampling for complex videos."
        )
        st.session_state["sampling_mode"] = sampling_mode

        st.divider()
        st.markdown(f"**VLM Provider:** `{settings.VLM_PROVIDER.upper()}`")
        st.markdown(f"**VLM Model:** `{settings.VLM_MODEL}`")
        st.markdown(f"**YOLO Threshold:** `{settings.YOLO_CONFIDENCE}`")
        st.markdown(f"**Max Video Duration:** `{settings.MAX_VIDEO_DURATION_SEC}s`")

        st.divider()
        st.markdown("### 🚫 Scope Restriction")
        st.info("Visual & Video Understanding Only. Audio analysis is disabled.")

    # Video Upload Section
    st.subheader("📤 Video Upload & Ingestion")

    uploaded_file = st.file_uploader(
        "Select or drop a video file (.mp4, .mov, .avi, .mkv)",
        type=["mp4", "mov", "avi", "mkv"],
        help="Upload a video to perform VLM analysis, tracking, temporal reasoning, and Q&A."
    )

    if uploaded_file is not None:
        save_path = settings.UPLOADS_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            memory = run_full_pipeline(save_path, sampling_mode=st.session_state["sampling_mode"])
            st.session_state["memory"] = memory
            st.session_state["metadata"] = memory.metadata
            st.session_state["video_path"] = str(save_path)
            st.success(f"✅ Video `{memory.metadata.filename}` processed with high accuracy!")
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            logger.error(f"Processing error: {e}", exc_info=True)

    # Render ingested video player & tabs if video exists
    if st.session_state["metadata"] is not None:
        st.divider()
        render_video_player(st.session_state["metadata"])

        memory: VideoMemory = st.session_state.get("memory")

        if memory:
            st.divider()

            # Global Metrics Row
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                render_metric_card("Duration", f"{memory.metadata.duration_sec:.1f}s", f"{memory.metadata.fps} FPS")
            with mcol2:
                render_metric_card("Scenes", str(len(memory.scenes)), "PySceneDetect")
            with mcol3:
                render_metric_card("Tracked Entities", str(len(memory.tracks)), "Spatial IoU Matcher")
            with mcol4:
                render_metric_card("Verified Events", str(len(memory.events)), "Event Engine")

            st.divider()

            # Quality Control Debug Summary Panel
            render_quality_control_panel(memory, len(memory.sampled_frames), memory.metadata.frame_count)

            st.divider()

            # Phase Tabs
            tabs = st.tabs([
                "📋 Overview",
                "🎬 Scenes",
                "🔍 Objects & People",
                "⚡ Events & Actions",
                "⏳ Timeline",
                "💡 Key Moments & Insights",
                "💬 Ask AI (Q&A)",
                "🔎 Semantic Search",
                "🎯 BBox Debugger",
                "📑 Report & Export"
            ])

            # Tab 0: Overview
            with tabs[0]:
                st.subheader("📝 Grounded Executive Summary")
                summary_level = st.radio("Select Summary Level:", ["Standard", "Quick", "Detailed", "Technical"], horizontal=True)
                key = summary_level.lower()
                st.write(memory.summary.get(key, memory.summary.get("standard", "")))

            # Tab 1: Scenes
            with tabs[1]:
                render_scenes_ui(memory.scenes)

            # Tab 2: Objects & People
            with tabs[2]:
                render_objects_and_tracks_ui(memory.tracks)

            # Tab 3: Events & Actions
            with tabs[3]:
                st.subheader("⚡ Verified Timed Events")
                for evt in memory.events:
                    conf_pct = int(evt.confidence * 100)
                    st.markdown(
                        f"**[{evt.start_time:.1f}s - {evt.end_time:.1f}s]** `{evt.event_type}` — {evt.description} *(Confidence: `{conf_pct}%`)*"
                    )

            # Tab 4: Timeline
            with tabs[4]:
                render_timeline_ui(memory.timeline)

            # Tab 5: Key Moments
            with tabs[5]:
                st.subheader("⭐ Key Video Moments")
                for idx, obs in enumerate(memory.frame_observations):
                    if obs.activities:
                        st.markdown(f"⭐ **[{obs.timestamp:.2f}s]** {', '.join(obs.activities)}")

            # Tab 6: Ask AI (Q&A)
            with tabs[6]:
                st.subheader("💬 Ask AI About This Video")
                question = st.text_input("Ask a question:", placeholder="e.g. When did the person enter? What objects were moved?")
                if question:
                    response = video_qa_engine.answer_question(memory, question)
                    st.markdown(f"### 🤖 Answer: {response.answer}")
                    st.markdown(f"**Confidence Score:** `{int(response.confidence * 100)}%`")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("#### 👁️ Direct Visual Observations")
                        for fact in response.observed_facts:
                            st.markdown(f"- {fact}")
                    with col_b:
                        st.markdown("#### ❓ Unknown / Unverified Aspects")
                        for unc in response.unknown_aspects:
                            st.markdown(f"- {unc}")

            # Tab 7: Semantic Search
            with tabs[7]:
                st.subheader("🔎 Semantic Video Search")
                search_query = st.text_input("Search visual moment:", placeholder="e.g. laptop on desk, person walking")
                if search_query:
                    results = semantic_search_engine.search(memory, search_query, top_k=5)
                    st.markdown(f"Found **{len(results)}** matching moments:")
                    for r in results:
                        st.markdown(f"⏱️ **[{r['timestamp']}s]** (Score: `{r['score']}`) — *{r['text']}*")

            # Tab 8: BBox Debugger
            with tabs[8]:
                render_bbox_debug_visualizer(memory.sampled_frames, memory.yolo_detections)

            # Tab 9: Report & Export
            with tabs[9]:
                st.subheader("📑 Generate & Export Comprehensive Report")
                reports = report_generator.generate_all_reports(memory)

                col_j, col_c, col_p = st.columns(3)
                with col_j:
                    with open(reports["json"], "rb") as f:
                        st.download_button("📥 Download JSON Report", f, file_name=reports["json"].name, mime="application/json")
                with col_c:
                    with open(reports["csv"], "rb") as f:
                        st.download_button("📥 Download CSV Events", f, file_name=reports["csv"].name, mime="text/csv")
                with col_p:
                    with open(reports["pdf"], "rb") as f:
                        st.download_button("📥 Download PDF Report", f, file_name=reports["pdf"].name, mime="application/pdf")
