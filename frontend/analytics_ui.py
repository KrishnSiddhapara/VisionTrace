import streamlit as st
from typing import List, Dict, Any
import pandas as pd
import cv2
from pathlib import Path

from models.schemas import TrackedObject, Scene, SampledFrame, YOLODetection

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


def render_quality_control_panel(memory_data: Dict[str, Any], sampled_count: int, total_frames: int) -> None:
    """Render Developer Debug & Processing Quality Control Panel."""
    st.subheader("🛠️ Video Processing Quality Control Panel")

    metadata = memory_data.metadata
    events = memory_data.events
    tracks = memory_data.tracks

    skipped_frames = max(0, total_frames - sampled_count)
    avg_conf = (sum(e.confidence for e in events) / len(events)) if events else 0.90

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Analyzed Frames", f"{sampled_count:,}", f"Skipped: {skipped_frames:,}")
    with col2:
        st.metric("PySceneDetect Scenes", len(memory_data.scenes))
    with col3:
        st.metric("Tracked Entities", len(tracks), "Spatial IoU Matcher")
    with col4:
        st.metric("Verified Events", len(events))
    with col5:
        st.metric("Avg Event Confidence", f"{int(avg_conf * 100)}%")


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
