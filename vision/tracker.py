import numpy as np
from typing import List, Dict, Any, Tuple
from models.schemas import SampledFrame, YOLODetection, TrackedObject, FrameObservation
from utils.logger import logger

CLASS_NORMALIZATION_MAP = {
    # Sports balls
    "sports ball": "sports ball",
    "soccer ball": "sports ball",
    "football": "sports ball",
    "basketball": "sports ball",
    "tennis ball": "sports ball",
    "baseball": "sports ball",
    "volleyball": "sports ball",
    "ball": "sports ball",

    # People
    "person": "person",
    "human": "person",
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "player": "person",

    # Bags & Luggage
    "backpack": "backpack",
    "bag": "backpack",
    "handbag": "backpack",
    "suitcase": "backpack",

    # Electronics
    "laptop": "laptop",
    "computer": "laptop",
    "cell phone": "cell phone",
    "phone": "cell phone",
    "mobile phone": "cell phone",
    "smartphone": "cell phone",

    # Vehicles
    "car": "car",
    "vehicle": "car",
    "automobile": "car",
}


UNNECESSARY_BACKGROUND_CLASSES = {
    "wall", "floor", "ceiling", "ground", "pavement", "road", "sky", "building",
    "window", "door", "curtain", "light", "shelf", "cabinet", "room", "background",
    "tree", "grass", "bench", "couch", "bed", "chair", "table", "desk", "dining table",
}


def is_main_object(cls_name: str, has_movement_or_interaction: bool = False) -> bool:
    """
    Check if an object category is a main, relevant visual object.
    Filters out unnecessary static background elements (wall, floor, table, etc.)
    unless they actively undergo movement or interaction in the frame.
    """
    if not cls_name:
        return False
    clean = cls_name.strip().lower()
    norm = CLASS_NORMALIZATION_MAP.get(clean, clean)
    if norm in UNNECESSARY_BACKGROUND_CLASSES or clean in UNNECESSARY_BACKGROUND_CLASSES:
        return has_movement_or_interaction
    return True


def normalize_class_name(cls_name: str) -> str:
    """Normalize object class names into canonical visual categories."""
    if not cls_name:
        return "object"
    clean = cls_name.strip().lower()
    return CLASS_NORMALIZATION_MAP.get(clean, clean)


def calculate_bbox_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    Returns IoU value between 0.0 and 1.0.
    """
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0.0, x2 - x1)
    inter_height = max(0.0, y2 - y1)
    inter_area = inter_width * inter_height

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return round(float(inter_area / union_area), 4)


def calculate_center_distance(box1: List[float], box2: List[float]) -> float:
    """Calculate Euclidean distance between bounding box centers."""
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
        return float("inf")
    cx1 = (box1[0] + box1[2]) / 2.0
    cy1 = (box1[1] + box1[3]) / 2.0
    cx2 = (box2[0] + box2[2]) / 2.0
    cy2 = (box2[1] + box2[3]) / 2.0
    return float(np.hypot(cx2 - cx1, cy2 - cy1))


class SpatialIoUTracker:
    """
    Tracks objects across sampled frames using class normalization, Bounding Box IoU,
    center-distance motion trajectory association, gap-tolerant occlusion handling,
    and state transition modeling.
    """

    def __init__(self, iou_threshold: float = 0.15, max_gap_sec: float = 8.0):
        self.iou_threshold = iou_threshold
        self.max_gap_sec = max_gap_sec

    def track_entities(
        self,
        sampled_frames: List[SampledFrame],
        frame_detections: Dict[str, List[YOLODetection]],
        frame_observations: List[FrameObservation]
    ) -> List[TrackedObject]:
        """
        Build persistent TrackedObject trajectories across frames.
        """
        active_tracks: Dict[str, TrackedObject] = {}
        class_counters: Dict[str, int] = {}
        obs_by_frame = {obs.frame_id: obs for obs in frame_observations}

        # Ensure frames are sorted chronologically
        sorted_frames = sorted(sampled_frames, key=lambda f: f.timestamp)

        for frame in sorted_frames:
            detections = frame_detections.get(frame.frame_id, [])
            vlm_obs = obs_by_frame.get(frame.frame_id)

            matched_track_ids = set()

            for det in detections:
                raw_cls = det.class_name
                norm_cls = normalize_class_name(raw_cls)
                det_bbox = det.bbox

                best_match_id = None
                best_score = 0.0

                # Search active tracks of same normalized class within gap tolerance
                for track_id, track in active_tracks.items():
                    track_norm_cls = normalize_class_name(track.canonical_name or track.object_type)
                    if track_norm_cls == norm_cls and track_id not in matched_track_ids:
                        gap = frame.timestamp - track.last_seen
                        if gap <= self.max_gap_sec:
                            last_pos = track.positions[-1]["bbox"] if track.positions else None
                            if last_pos:
                                iou = calculate_bbox_iou(det_bbox, last_pos)
                                dist = calculate_center_distance(det_bbox, last_pos)
                                box_size = max(last_pos[2] - last_pos[0], last_pos[3] - last_pos[1], 30.0)
                                norm_dist = dist / box_size

                                score = 0.0
                                if iou >= self.iou_threshold:
                                    score = 0.6 + 0.4 * iou
                                elif norm_dist <= 3.0 or dist <= 150.0:
                                    # Motion / displacement match for moving objects (e.g. ball, walking person)
                                    score = max(0.05, 0.5 - 0.1 * norm_dist)

                                if score > 0.0 and score > best_score:
                                    best_score = score
                                    best_match_id = track_id

                if best_match_id:
                    # Match found -> update existing track trajectory
                    track = active_tracks[best_match_id]
                    det.track_id = best_match_id
                    matched_track_ids.add(best_match_id)
                    track.last_seen = frame.timestamp
                    track.detection_count += 1
                    # Update average confidence
                    track.avg_confidence = round(
                        (track.avg_confidence * (track.detection_count - 1) + det.confidence) / track.detection_count,
                        3
                    )
                    track.positions.append({"timestamp": frame.timestamp, "bbox": det_bbox})
                    track.state_history.append({"timestamp": frame.timestamp, "state": "tracked_position"})
                else:
                    # No match -> spawn new track with entity history
                    display_cls = "Ball" if norm_cls == "sports ball" else norm_cls.capitalize()
                    if display_cls not in class_counters:
                        class_counters[display_cls] = 1

                    new_track_id = f"{display_cls} #{class_counters[display_cls]}"
                    class_counters[display_cls] += 1
                    det.track_id = new_track_id
                    matched_track_ids.add(new_track_id)

                    active_tracks[new_track_id] = TrackedObject(
                        track_id=new_track_id,
                        object_type=raw_cls,
                        canonical_name=norm_cls,
                        first_seen=frame.timestamp,
                        last_seen=frame.timestamp,
                        positions=[{"timestamp": frame.timestamp, "bbox": det_bbox}],
                        activities=[],
                        interactions=[],
                        lifecycle_events=["appeared"],
                        state_history=[{"timestamp": frame.timestamp, "state": "appeared"}],
                        detection_count=1,
                        avg_confidence=det.confidence,
                    )

                # Attach activities & interactions from VLM observations
                assigned_id = det.track_id
                if vlm_obs and assigned_id in active_tracks:
                    track = active_tracks[assigned_id]
                    for act in vlm_obs.activities:
                        if act not in track.activities:
                            track.activities.append(act)
                    for inter in vlm_obs.interactions:
                        if inter not in track.interactions:
                            track.interactions.append(inter)

        tracked_list = list(active_tracks.values())

        # Determine object lifecycle events and state transitions
        for trk in tracked_list:
            if len(trk.positions) > 1:
                p_first = trk.positions[0]["bbox"]
                p_last = trk.positions[-1]["bbox"]
                center_first = ((p_first[0] + p_first[2]) / 2, (p_first[1] + p_first[3]) / 2)
                center_last = ((p_last[0] + p_last[2]) / 2, (p_last[1] + p_last[3]) / 2)
                dist = float(np.hypot(center_last[0] - center_first[0], center_last[1] - center_first[1]))
                if dist > 40.0 and "moved" not in trk.lifecycle_events:
                    trk.lifecycle_events.append("moved")
                    trk.state_history.append({"timestamp": trk.last_seen, "state": "moved"})

            # Check if entity exited visible frame before video end
            if sorted_frames and (sorted_frames[-1].timestamp - trk.last_seen) > 5.0:
                if "exited" not in trk.lifecycle_events:
                    trk.lifecycle_events.append("exited")
                    trk.state_history.append({"timestamp": trk.last_seen, "state": "exited"})

        logger.info(f"Spatial IoU Tracker built trajectories for {len(tracked_list)} distinct entities.")
        return tracked_list


object_tracker = SpatialIoUTracker()
