# pyrefly: ignore [missing-import]
import streamlit as st
from typing import List, Dict, Any
import pandas as pd
# pyrefly: ignore [missing-import]
import cv2
from pathlib import Path

from models.schemas import TrackedObject, Scene, SampledFrame, YOLODetection

def render_final_summary_ui(memory: Any) -> None:
    """
    Render Final Video Summary containing three primary sections:
    1. OBJECTS
    2. PEOPLE
    3. FINAL DESCRIPTION
    with evidence keyframe previews.
    """
    final_sum = getattr(memory, "final_summary", None)
    
    # 1. SECTION 1 — OBJECTS
    st.markdown("## 📦 OBJECTS")
    objects = final_sum.objects if final_sum and final_sum.objects else []
    
    if not objects:
        st.info("No distinct objects detected with sufficient visual evidence.")
    else:
        for idx, obj in enumerate(objects, start=1):
            with st.expander(f"📦 Object {idx}: {obj.name} (Confidence: {int(obj.confidence * 100)}%)", expanded=True):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.markdown(f"**Name:** `{obj.name}`")
                    st.markdown(f"**First Seen:** `{obj.first_seen}`")
                    st.markdown(f"**Last Seen:** `{obj.last_seen}`")
                    st.markdown(f"**Confidence:** `{int(obj.confidence * 100)}%`")
                with col_b:
                    st.markdown(f"**Description:** {obj.description}")
                    st.markdown(f"**Movement:** {obj.movement}")
                    if obj.state_changes:
                        st.markdown(f"**State Changes:** {', '.join(obj.state_changes)}")

    st.divider()

    # 2. SECTION 2 — PEOPLE
    st.markdown("## 👤 PEOPLE")
    people = final_sum.people if final_sum and final_sum.people else []

    if not people:
        st.info("No person entities detected with sufficient visual evidence.")
    else:
        for p in people:
            with st.expander(f"👤 {p.temporary_id} (Confidence: {int(p.confidence * 100)}%)", expanded=True):
                col_x, col_y = st.columns([1, 2])
                with col_x:
                    st.markdown(f"**Temporary ID:** `{p.temporary_id}`")
                    st.markdown(f"**First Seen:** `{p.first_seen}`")
                    st.markdown(f"**Last Seen:** `{p.last_seen}`")
                    st.markdown(f"**Confidence:** `{int(p.confidence * 100)}%`")
                with col_y:
                    st.markdown(f"**Description:** {p.description}")
                    st.markdown(f"**Activities:** {', '.join(p.activities)}")
                    st.markdown(f"**Movements:** {', '.join(p.movements)}")

    st.divider()

    # 3. SECTION 3 — FINAL DESCRIPTION
    st.markdown("## 📝 FINAL DESCRIPTION")
    final_desc = final_sum.final_description if final_sum and final_sum.final_description else memory.summary.get("standard", "No description available.")
    
    st.markdown(
        f"""
        <div style="background-color: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 10px; border-left: 4px solid #F16663; line-height: 1.6; font-size: 1.05rem;">
        {final_desc}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # 4. Evidence Frame Previews
    st.markdown("### 📷 Evidence Movement Keyframe Previews")
    if memory.sampled_frames:
        cols = st.columns(min(4, len(memory.sampled_frames)))
        for idx, sf in enumerate(memory.sampled_frames[:8]):
            col_target = cols[idx % len(cols)]
            img_p = Path(sf.path)
            if img_p.exists():
                col_target.image(
                    str(img_p),
                    caption=f"⏱️ [{sf.timestamp:.2f}s]\nReason: {sf.selection_reason}\nMotion: {sf.motion_score:.2f}",
                    use_container_width=True
                )


def render_scenes_ui(scenes: List[Scene]) -> None:
    """Render detected video scenes table & metadata."""
    st.subheader("🎬 Detected PySceneDetect Scenes")
    if not scenes:
        st.info("No scenes detected.")
        return

    data = []
    for sc in scenes:
        data.append({
            "Scene ID": sc.scene_id,
            "Start Frame": sc.start_frame,
            "End Frame": sc.end_frame,
            "Start Time": f"{sc.start_time:.2f}s",
            "End Time": f"{sc.end_time:.2f}s",
            "Duration": f"{sc.duration:.2f}s",
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)


def render_objects_and_tracks_ui(tracks: List[TrackedObject]) -> None:
    """Render tracked objects and entity histories."""
    st.subheader("🔍 Tracked Entities & Spatial Trajectories")

    if not tracks:
        st.info("No tracked entities found.")
        return

    data = []
    for trk in tracks:
        data.append({
            "Track ID": trk.track_id,
            "Entity Type": trk.object_type.capitalize(),
            "First Seen": f"{trk.first_seen:.2f}s",
            "Last Seen": f"{trk.last_seen:.2f}s",
            "Observed Frames": len(trk.positions),
            "Lifecycle Events": ", ".join(trk.lifecycle_events) if trk.lifecycle_events else "None",
            "Activities": ", ".join(trk.activities) if trk.activities else "Observed in scene",
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)


def render_quality_control_panel(memory_data: Any, sampled_count: int, total_frames: int) -> None:
    """Render Developer Debug & Processing Quality Control Panel."""
    st.subheader("🛠️ Video Processing Quality Control Panel")

    events = memory_data.events
    tracks = memory_data.tracks

    skipped_frames = max(0, total_frames - sampled_count)
    avg_conf = (sum(e.confidence for e in events) / len(events)) if events else 0.90
    confirmed_count = len([e for e in events if getattr(e, "evidence_level", "CONFIRMED") == "CONFIRMED"])

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Analyzed Frames", f"{sampled_count:,}", f"Skipped: {skipped_frames:,}")
    with col2:
        st.metric("PySceneDetect Scenes", len(memory_data.scenes))
    with col3:
        st.metric("Tracked Entities", len(tracks), "Gap-Tolerant IoU Matcher")
    with col4:
        st.metric("Verified Events", len(events), f"{confirmed_count} Confirmed")
    with col5:
        st.metric("Avg Event Confidence", f"{int(avg_conf * 100)}%")


def render_movement_frames_ui(sampled_frames: List[SampledFrame]) -> None:
    """Render Movement Frames Viewer with OpenCV Motion & Selection Reason Metadata."""
    st.subheader("🎬 OpenCV Selected Movement & Change Frames")
    st.info("Frames selected by OpenCV change detection for VLM ingestion")

    if not sampled_frames:
        st.warning("No movement frames selected.")
        return

    # Data Table
    table_data = []
    for sf in sampled_frames:
        table_data.append({
            "Frame ID": sf.frame_id,
            "Timestamp": f"{sf.timestamp:.2f}s",
            "Selection Reason": sf.selection_reason,
            "Motion Score": f"{sf.motion_score:.2f}",
            "Change Score": f"{sf.change_score:.2f}",
            "Motion Area": f"{int(sf.motion_area * 100)}%",
            "Quality Score": f"{sf.quality_score:.2f}",
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)

    st.divider()

    # Image Inspector
    frame_map = {f"⏱️ [{sf.timestamp:.2f}s] {sf.selection_reason} (Motion: {sf.motion_score:.2f})": sf for sf in sampled_frames}
    selected_label = st.selectbox("Inspect Movement Frame Details & Visual Image:", list(frame_map.keys()))

    if selected_label:
        sf = frame_map[selected_label]
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"**Frame ID:** `{sf.frame_id}`")
            st.markdown(f"**Timestamp:** `{sf.timestamp:.2f}s`")
            st.markdown(f"**Selection Reason:** `{sf.selection_reason}`")
            st.markdown(f"**Motion Score:** `{sf.motion_score:.3f}`")
            st.markdown(f"**Change Score:** `{sf.change_score:.3f}`")
            st.markdown(f"**Motion Area:** `{int(sf.motion_area * 100)}%`")
        with c2:
            img_path = Path(sf.path)
            if img_path.exists():
                st.image(str(img_path), caption=f"{sf.frame_id} — {sf.selection_reason} ({sf.timestamp}s)", use_container_width=True)


def render_developer_accuracy_dashboard(memory: Any) -> None:
    """Render Developer Accuracy Metrics Dashboard."""
    st.subheader("📊 Developer Accuracy & Pipeline Metrics")
    st.info("High-Accuracy Pipeline Telemetry & Evidence Breakdown")

    metrics = memory.developer_metrics
    events = memory.events
    frame_obs = memory.frame_observations
    total_vid_frames = memory.metadata.frame_count

    st.markdown("### ⚡ OpenCV Change-Driven Frame Reduction Telemetry")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("Total Video Frames", f"{total_vid_frames:,}")
    with p2:
        st.metric("Candidate Motion Frames", len(memory.sampled_frames) * 3, "OpenCV Scanned")
    with p3:
        st.metric("Selected Keyframes", len(memory.sampled_frames), f"Saved: {total_vid_frames - len(memory.sampled_frames):,}")
    with p4:
        st.metric("VLM Analyzed Calls", len([o for o in frame_obs if o.is_analyzed]))

    st.divider()

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Frames Sampled", len(memory.sampled_frames))
    with m2:
        st.metric("Frames Analyzed", len([o for o in frame_obs if o.is_analyzed]))
    with m3:
        st.metric("Unanalyzed/Skipped", len([o for o in frame_obs if not o.is_analyzed]))
    with m4:
        st.metric("VLM Retries/Failures", f"{metrics.vlm_retries if metrics else 0}/{metrics.vlm_failures if metrics else 0}")
    with m5:
        st.metric("Tracked Entities", len(memory.tracks))
    with m6:
        st.metric("Verified Events", len(events))

    # Breakdown of Evidence Levels
    st.markdown("### 🏆 Event Verification & Evidence Levels")
    level_counts = {
        "CONFIRMED": len([e for e in events if getattr(e, "evidence_level", "CONFIRMED") == "CONFIRMED"]),
        "PROBABLE": len([e for e in events if getattr(e, "evidence_level", "") == "PROBABLE"]),
        "UNCERTAIN": len([e for e in events if getattr(e, "evidence_level", "") == "UNCERTAIN"]),
        "REJECTED": metrics.rejected_events_count if metrics else 0,
    }

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.success(f"🟢 CONFIRMED: {level_counts['CONFIRMED']}")
    with c2:
        st.info(f"🔵 PROBABLE: {level_counts['PROBABLE']}")
    with c3:
        st.warning(f"🟡 UNCERTAIN: {level_counts['UNCERTAIN']}")
    with c4:
        st.error(f"🔴 REJECTED: {level_counts['REJECTED']}")


def render_bbox_debug_visualizer(sampled_frames: List[SampledFrame], frame_detections: Dict[str, List[YOLODetection]]) -> None:
    """Render Bounding Box Debug Overlay Viewer for selected sampled frames."""
    st.subheader("🎯 Bounding Box & Track ID Debug Overlay Visualizer")

    if not sampled_frames:
        st.info("No sampled frames available for debug rendering.")
        return

    frame_options = {f"{sf.frame_id} (Timestamp: {sf.timestamp}s)": sf for sf in sampled_frames}
    selected_key = st.selectbox("Select Sampled Frame to Inspect Bounding Boxes:", list(frame_options.keys()))

    if selected_key:
        sf = frame_options[selected_key]
        img_path = Path(sf.path)

        if img_path.exists():
            image = cv2.imread(str(img_path))
            dets = frame_detections.get(sf.frame_id, [])

            # Draw bounding boxes and labels
            for d in dets:
                bbox = d.bbox
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    label = f"{d.track_id or d.class_name} ({int(d.confidence*100)}%)"

                    # Draw BBox rectangle & label
                    cv2.rectangle(image, (x1, y1), (x2, y2), (241, 102, 99), 2)
                    cv2.putText(image, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption=f"{sf.frame_id} - Detections: {len(dets)}", use_container_width=True)
