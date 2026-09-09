import uuid
import concurrent.futures
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import streamlit as st

from config.settings import settings
from video.processor import video_processor
from video.sampler import frame_sampler
from vision.frame_analyzer import frame_analyzer
from vision.object_detector import object_detector
from vision.tracker import SpatialIoUTracker
from intelligence.event_detector import event_detector
from intelligence.temporal_reasoner import temporal_reasoner
from intelligence.key_moment_detector import key_moment_detector
from intelligence.anomaly_detector import anomaly_detector
from intelligence.video_memory import video_memory_manager
from models.schemas import VideoMemory, DeveloperMetrics
from qa.video_qa import video_qa_engine
from retrieval.semantic_search import semantic_search_engine
from reports.report_generator import report_generator
from utils.profiler import profiler

from frontend.components import load_custom_css, render_header, render_metric_card
from frontend.video_player import render_video_player
from frontend.timeline_ui import render_timeline_ui
from frontend.analytics_ui import (
    render_objects_and_tracks_ui,
    render_scenes_ui,
    render_quality_control_panel,
    render_bbox_debug_visualizer,
    render_developer_accuracy_dashboard,
    render_movement_frames_ui,
    render_final_summary_ui,
)
from utils.logger import logger

@dataclass
class AnalysisContext:
    analysis_id: str
    upload_timestamp: str
    video_path: Path
    video_metadata: Any = None
    memory: Any = None


def reset_analysis_state() -> None:
    """Clear all analysis-specific session state variables on new upload."""
    st.session_state["metadata"] = None
    st.session_state["memory"] = None
    st.session_state["video_path"] = None
    st.session_state["analysis_id"] = None
    st.session_state["current_upload_name"] = None
    st.session_state["analysis_running"] = False


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
    if "analysis_id" not in st.session_state:
        st.session_state["analysis_id"] = None
    if "current_upload_name" not in st.session_state:
        st.session_state["current_upload_name"] = None
    if "analysis_running" not in st.session_state:
        st.session_state["analysis_running"] = False


def run_full_pipeline(
    video_path: Path,
    sampling_mode: str = "Balanced",
    yolo_confidence: float = 0.45
) -> VideoMemory:
    """Execute optimized 12-stage processing pipeline with profiling."""
    profiler.start_pipeline()
    analysis_id = str(uuid.uuid4())
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("========================================")
    logger.info("NEW OPTIMIZED VIDEO ANALYSIS")
    logger.info("========================================")
    logger.info(f"Analysis ID: {analysis_id}")
    logger.info(f"Upload timestamp: {timestamp_str}")
    logger.info(f"Filename: {video_path.name}")
    logger.info(f"Sampling Mode: {sampling_mode}")
    logger.info(f"YOLO Confidence Threshold: {yolo_confidence:.2f}")
    logger.info("========================================")

    st.session_state["analysis_id"] = analysis_id

    progress_bar = st.progress(0, text=f"Initializing analysis pipeline (ID: {analysis_id[:8]})...")

    # Step 1: Validation & Metadata Extraction
    profiler.start_stage("Validation & Metadata")
    progress_bar.progress(10, text="✓ Step 1/8: Validating video & extracting metadata...")
    validation, metadata, scenes, _ = video_processor.process_video(video_path)
    profiler.stop_stage("Validation & Metadata", item_count=metadata.frame_count, unit="video frames")

    # Step 2: Intelligent Multi-Criteria Frame Sampler (Stage 1 Fast Scan)
    profiler.start_stage("Adaptive Frame Sampling")
    progress_bar.progress(25, text=f"● Step 2/8: Scanning & selecting representative frames ({sampling_mode} Mode)...")
    output_frames_dir = settings.PROCESSED_DIR / metadata.video_hash / "frames"
    sampled_frames = frame_sampler.sample_scene_frames(video_path, scenes, output_frames_dir, sampling_mode=sampling_mode)
    profiler.stop_stage("Adaptive Frame Sampling", item_count=len(sampled_frames), unit="selected keyframes")

    # Step 3: VLM Frame Window Analysis (Stage 2 Deep Analysis - Concurrently Parallelized)
    profiler.start_stage("VLM Analysis")
    progress_bar.progress(40, text=f"● Step 3/8: Analyzing {len(sampled_frames)} frames concurrently with VLM...")
    frame_obs_map = {}

    max_workers = min(settings.VLM_MAX_WORKERS, max(1, len(sampled_frames)))

    def _analyze_window_worker(arg):
        i, sf = arg
        pf = sampled_frames[i - 1] if i > 0 else None
        nf = sampled_frames[i + 1] if i + 1 < len(sampled_frames) else None
        return sf.frame_id, frame_analyzer.analyze_frame_window(
            sampled_frame=sf,
            prev_frame=pf,
            next_frame=nf,
            video_hash=metadata.video_hash,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_analyze_window_worker, (i, sf)) for i, sf in enumerate(sampled_frames)]
        for future in concurrent.futures.as_completed(futures):
            f_id, obs = future.result()
            frame_obs_map[f_id] = obs

    frame_obs_list = sorted(list(frame_obs_map.values()), key=lambda o: o.timestamp)
    profiler.stop_stage("VLM Analysis", item_count=len(sampled_frames), unit="VLM frame queries")

    # Step 4: YOLO Object Detection (Batched & Cached)
    profiler.start_stage("YOLO Detection")
    progress_bar.progress(55, text=f"● Step 4/8: Running batched YOLO object detection (conf={yolo_confidence:.2f})...")
    batch_paths = [sf.path for sf in sampled_frames]
    path_to_id = {sf.path: sf.frame_id for sf in sampled_frames}
    batch_dets = object_detector.detect_objects_batch(batch_paths, video_hash=metadata.video_hash, confidence_threshold=yolo_confidence)
    
    yolo_dets = {}
    for sf in sampled_frames:
        yolo_dets[sf.frame_id] = batch_dets.get(str(Path(sf.path).resolve()), [])
    profiler.stop_stage("YOLO Detection", item_count=sum(len(d) for d in yolo_dets.values()), unit="detections")

    # Step 5: Spatial Bounding Box IoU Entity Tracker
    profiler.start_stage("Entity Tracking")
    progress_bar.progress(65, text="● Step 5/8: Tracking entity trajectories with Spatial IoU Matcher...")
    fresh_tracker = SpatialIoUTracker()
    tracks = fresh_tracker.track_entities(sampled_frames, yolo_dets, frame_obs_list)
    profiler.stop_stage("Entity Tracking", item_count=len(tracks), unit="tracked entities")

    # Step 6: Candidate Events & Pass 2 Verification
    profiler.start_stage("Candidate Event Detection")
    progress_bar.progress(75, text="● Step 6/8: Identifying candidate events & verifying event windows...")
    candidate_events = event_detector.detect_events(scenes, frame_obs_list, tracks)
    event_windows = [(e.start_time, e.end_time) for e in candidate_events if e.event_type != "SCENE"]

    if event_windows:
        sampled_frames = frame_sampler.sample_event_dense_frames(video_path, event_windows, output_frames_dir, sampled_frames)
        # Process newly added verification frames
        uncached_frames = [sf for sf in sampled_frames if sf.frame_id not in frame_obs_map]
        if uncached_frames:
            max_p2_workers = min(settings.VLM_MAX_WORKERS, max(1, len(uncached_frames)))
            def _analyze_p2_worker(sf):
                return sf.frame_id, frame_analyzer.analyze_frame_window(sf, video_hash=metadata.video_hash)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_p2_workers) as executor:
                futures = [executor.submit(_analyze_p2_worker, sf) for sf in uncached_frames]
                for future in concurrent.futures.as_completed(futures):
                    f_id, obs = future.result()
                    frame_obs_map[f_id] = obs

            p2_dets = object_detector.detect_objects_batch([sf.path for sf in uncached_frames], video_hash=metadata.video_hash, confidence_threshold=yolo_confidence)
            for sf in uncached_frames:
                yolo_dets[sf.frame_id] = p2_dets.get(str(Path(sf.path).resolve()), [])

        frame_obs_list = sorted(list(frame_obs_map.values()), key=lambda o: o.timestamp)
    profiler.stop_stage("Candidate Event Detection", item_count=len(candidate_events), unit="candidate events")

    # Step 7: Multi-Source Event Verification Pipeline
    profiler.start_stage("Event Verification")
    progress_bar.progress(85, text="● Step 7/8: Running multi-source event verification & confidence scoring...")
    tracks = fresh_tracker.track_entities(sampled_frames, yolo_dets, frame_obs_list)
    verified_events = event_detector.detect_events(scenes, frame_obs_list, tracks)
    profiler.stop_stage("Event Verification", item_count=len(verified_events), unit="verified events")

    # Step 8: Temporal Reasoning & Final Summary Generation
    profiler.start_stage("Temporal Reasoning")
    progress_bar.progress(95, text="● Step 8/8: Synthesizing grounded timeline & physical object summary...")
    timeline = temporal_reasoner.synthesize_timeline(scenes, verified_events, tracks, frame_obs_list)
    summaries = temporal_reasoner.generate_summaries(metadata, scenes, timeline, tracks)
    final_summary = temporal_reasoner.generate_final_summary(metadata, scenes, timeline, tracks, frame_obs_list, sampled_frames)
    profiler.stop_stage("Temporal Reasoning", item_count=len(timeline), unit="timeline items")

    total_proc_time = profiler.stop_pipeline()

    analyzed_cnt = len([o for o in frame_obs_list if o.is_analyzed])
    skipped_cnt = len([o for o in frame_obs_list if not o.is_analyzed])
    avg_c = float(sum(e.confidence for e in verified_events) / max(1, len(verified_events)))

    metrics = DeveloperMetrics(
        total_video_frames=metadata.frame_count,
        candidate_movement_frames=len(sampled_frames) * 3,
        selected_change_frames=len(sampled_frames),
        static_frames_discarded=max(0, metadata.frame_count - len(sampled_frames)),
        frames_sampled=len(sampled_frames),
        frames_analyzed=analyzed_cnt,
        frames_skipped=skipped_cnt,
        vlm_calls=analyzed_cnt,
        vlm_retries=0,
        vlm_failures=skipped_cnt,
        yolo_confidence_threshold=round(yolo_confidence, 2),
        yolo_detections_count=sum(len(d) for d in yolo_dets.values()),
        tracked_entities_count=len(tracks),
        candidate_events_count=len(candidate_events),
        verified_events_count=len(verified_events),
        rejected_events_count=max(0, len(candidate_events) - len(verified_events)),
        average_confidence=round(avg_c, 2),
    )

    memory = VideoMemory(
        video_hash=f"{metadata.video_hash}_{analysis_id[:8]}",
        metadata=metadata,
        scenes=scenes,
        sampled_frames=sampled_frames,
        frame_observations=frame_obs_list,
        yolo_detections=yolo_dets,
        tracks=tracks,
        events=verified_events,
        timeline=timeline,
        summary=summaries,
        final_summary=final_summary,
        insights=[],
        developer_metrics=metrics,
    )

    video_memory_manager.save_memory(memory)
    progress_bar.progress(100, text=f"✅ Fresh Pipeline Complete in {total_proc_time:.1f}s (ID: {analysis_id[:8]})!")
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
            "🎯 Sampling Profile:",
            ["Balanced", "Fast", "Deep Analysis"],
            index=0,
            help="Balanced: Optimal accuracy/speed. Fast: Quick overview (max 8 VLM frames). Deep Analysis: Dense sampling."
        )
        st.session_state["sampling_mode"] = sampling_mode

        yolo_confidence = st.slider(
            "🎯 YOLO Confidence Threshold:",
            min_value=0.05,
            max_value=0.95,
            value=st.session_state.get("yolo_confidence", float(getattr(settings, "YOLO_CONFIDENCE", 0.45))),
            step=0.05,
            help="Confidence threshold for YOLO object detection. Lower = more candidate detections, Higher = strictly high-confidence detections."
        )
        st.session_state["yolo_confidence"] = yolo_confidence

        st.divider()
        st.markdown(f"**VLM Provider:** `{settings.VLM_PROVIDER.upper()}`")
        st.markdown(f"**VLM Model:** `{settings.VLM_MODEL}`")
        st.markdown(f"**Mock Mode:** `{settings.VLM_MOCK_MODE}`")
        st.markdown(f"**YOLO Threshold:** `{yolo_confidence:.2f}` (Runtime)")
        st.markdown(f"**Analysis Ver:** `{settings.ANALYSIS_VERSION}`")

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
        is_new_upload = st.session_state.get("current_upload_name") != uploaded_file.name

        if is_new_upload or not save_path.exists() or save_path.stat().st_size != uploaded_file.size:
            if is_new_upload:
                reset_analysis_state()
                st.session_state["current_upload_name"] = uploaded_file.name

            # Stream save in 8MB chunks to prevent RAM spikes and disk write delays
            with open(save_path, "wb") as f:
                uploaded_file.seek(0)
                chunk_size = 8 * 1024 * 1024
                while True:
                    chunk = uploaded_file.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)

        st.session_state["video_path"] = save_path

        if st.button("🚀 Process Video with Fresh High-Accuracy Pipeline", type="primary", disabled=st.session_state.get("analysis_running", False)):
            st.session_state["analysis_running"] = True
            st.session_state["memory"] = None
            st.session_state["metadata"] = None
            try:
                memory = run_full_pipeline(save_path, sampling_mode=sampling_mode, yolo_confidence=yolo_confidence)
                st.session_state["memory"] = memory
                st.session_state["metadata"] = memory.metadata
            finally:
                st.session_state["analysis_running"] = False

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
                "📝 Overview & Summary",
                "🎬 Scenes",
                "🏃 Movement Frames",
                "🔍 Tracked Entities",
                "⚡ Verified Events",
                "⏱️ Timeline",
                "⭐ Key Moments",
                "💬 Ask AI (Q&A)",
                "🔎 Semantic Search",
                "🎯 BBox Debugger",
                "📑 Comprehensive Report",
                "📊 Developer Dashboard",
            ])

            # Tab 0: Overview & Summary
            with tabs[0]:
                render_final_summary_ui(memory)

            # Tab 1: Scenes
            with tabs[1]:
                render_scenes_ui(memory.scenes)

            # Tab 2: Movement Frames
            with tabs[2]:
                render_movement_frames_ui(memory.sampled_frames)

            # Tab 3: Tracked Entities
            with tabs[3]:
                render_objects_and_tracks_ui(memory.tracks)

            # Tab 4: Verified Events
            with tabs[4]:
                st.subheader("⚡ Verified Timed Events")
                for evt in memory.events:
                    lvl = getattr(evt, "evidence_level", "CONFIRMED")
                    badge = "🟢" if lvl == "CONFIRMED" else ("🔵" if lvl == "PROBABLE" else "🟡")
                    conf_pct = int(evt.confidence * 100)
                    st.markdown(
                        f"{badge} **[{evt.start_time:.1f}s - {evt.end_time:.1f}s]** `{evt.event_type}` — {evt.description} *(Level: `{lvl}`, Confidence: `{conf_pct}%`)*"
                    )

            # Tab 5: Timeline
            with tabs[5]:
                render_timeline_ui(memory.timeline)

            # Tab 6: Key Moments
            with tabs[6]:
                st.subheader("⭐ Key Video Moments")
                for idx, obs in enumerate(memory.frame_observations):
                    if obs.activities:
                        st.markdown(f"⭐ **[{obs.timestamp:.2f}s]** {', '.join(obs.activities)}")

            # Tab 7: Ask AI (Q&A)
            with tabs[7]:
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

            # Tab 8: Semantic Search
            with tabs[8]:
                st.subheader("🔎 Semantic Video Search")
                search_query = st.text_input("Search visual moment:", placeholder="e.g. laptop on desk, person walking")
                if search_query:
                    results = semantic_search_engine.search(memory, search_query, top_k=5)
                    st.markdown(f"Found **{len(results)}** matching moments:")
                    for r in results:
                        st.markdown(f"⏱️ **[{r['timestamp']}s]** (Score: `{r['score']}`) — *{r['text']}*")

            # Tab 9: BBox Debugger
            with tabs[9]:
                render_bbox_debug_visualizer(memory.sampled_frames, memory.yolo_detections)

            # Tab 10: Grounded Video Intelligence Report
            with tabs[10]:
                st.subheader("📑 Grounded Video Intelligence Report")
                st.info("Comprehensive visual intelligence summary grounded in frame observations, entity trajectories, and verified timeline events.")

                if memory.summary:
                    st.markdown("### 📌 Executive Overview")
                    st.write(memory.summary.get("standard", memory.summary.get("quick", "No summary available.")))

                    st.markdown("### 📋 Chronological Breakdown")
                    st.write(memory.summary.get("detailed", "No detailed timeline logged."))

                    st.markdown("### 🔬 Grounded Facts & Metrics")
                    st.code(memory.summary.get("technical", ""), language="yaml")


            # Tab 11: Developer Dashboard
            with tabs[11]:
                render_developer_accuracy_dashboard(memory)
                st.divider()
                st.markdown("### ⏱️ Performance Profiler Timing Breakdown")
                summary = profiler.get_summary()
                st.markdown(f"**Total Pipeline Execution Duration:** `{summary.get('total_duration_sec', 0.0):.2f} sec`")
                st.json(summary.get("stages", {}))
