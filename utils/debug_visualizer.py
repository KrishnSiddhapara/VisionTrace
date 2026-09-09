import cv2
from pathlib import Path
from typing import List, Dict, Any
from models.schemas import SampledFrame, YOLODetection, TrackedObject
from utils.logger import logger

def render_debug_tracking_frames(
    sampled_frames: List[SampledFrame],
    frame_detections: Dict[str, List[YOLODetection]],
    tracks: List[TrackedObject],
    person_entities: List[Any],
    output_dir: Path,
) -> None:
    """
    Render annotated debug frames showing bounding boxes, tracker ID, canonical Person ID,
    confidence score, and track state when DEBUG_PERSON_TRACKING is enabled.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map track_id to canonical person_id
    track_to_person: Dict[str, str] = {}
    for pe in person_entities:
        p_id = getattr(pe, "person_id", pe.get("person_id") if isinstance(pe, dict) else "Person #?")
        t_ids = getattr(pe, "track_ids", pe.get("track_ids", []) if isinstance(pe, dict) else [])
        for tid in t_ids:
            track_to_person[tid] = p_id

    # Map track_id to TrackedObject for state lookups
    track_map = {t.track_id: t for t in tracks}

    for sf in sampled_frames:
        if not sf.path or not Path(sf.path).exists():
            continue

        img = cv2.imread(sf.path)
        if img is None:
            continue

        dets = frame_detections.get(sf.frame_id, [])

        for det in dets:
            if not det.bbox or len(det.bbox) < 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in det.bbox[:4]]
            cls_name = det.class_name
            is_person = cls_name.lower() in ("person", "people", "human", "man", "woman", "child", "kid", "player")

            tid = det.track_id or "Unassigned"
            pid = track_to_person.get(tid, "N/A")
            trk_obj = track_map.get(tid)
            state = trk_obj.track_state if trk_obj else "TENTATIVE"

            if is_person:
                color = (0, 255, 0) if state == "CONFIRMED" else (0, 165, 255)  # Green for confirmed, Orange for tentative
            else:
                color = (255, 0, 0)  # Blue for objects

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{tid} | {pid} | {det.confidence:.2f} | {state}"
            cv2.putText(
                img, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
            cv2.putText(
                img, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        out_path = output_dir / f"debug_{sf.frame_id}.jpg"
        cv2.imwrite(str(out_path), img)

    logger.info(f"Rendered {len(sampled_frames)} debug tracking frames to {output_dir}")
