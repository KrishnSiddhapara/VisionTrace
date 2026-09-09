import numpy as np
from typing import List, Dict, Any, Tuple, Optional

def solve_linear_sum_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    1-to-1 Hungarian linear sum assignment matching solver.
    Tries scipy.optimize.linear_sum_assignment if available; falls back to exact/greedy 1-to-1 solver.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost_matrix)
    except Exception:
        if cost_matrix.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
            
        num_rows, num_cols = cost_matrix.shape
        row_indices = []
        col_indices = []
        
        pairs = []
        for r in range(num_rows):
            for c in range(num_cols):
                pairs.append((cost_matrix[r, c], r, c))
        pairs.sort(key=lambda x: x[0])
        
        assigned_rows = set()
        assigned_cols = set()
        for cost, r, c in pairs:
            if r not in assigned_rows and c not in assigned_cols:
                assigned_rows.add(r)
                assigned_cols.add(c)
                row_indices.append(r)
                col_indices.append(c)
                if len(assigned_rows) == min(num_rows, num_cols):
                    break
                    
        return np.array(row_indices, dtype=int), np.array(col_indices, dtype=int)

from config.settings import settings
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

    # People & Synonyms
    "person": "person",
    "people": "person",
    "human": "person",
    "humans": "person",
    "man": "person",
    "men": "person",
    "woman": "person",
    "women": "person",
    "boy": "person",
    "boys": "person",
    "girl": "person",
    "girls": "person",
    "child": "person",
    "children": "person",
    "kid": "person",
    "kids": "person",
    "toddler": "person",
    "toddlers": "person",
    "baby": "person",
    "babies": "person",
    "infant": "person",
    "infants": "person",
    "youth": "person",
    "player": "person",
    "players": "person",
    "pedestrian": "person",
    "pedestrians": "person",

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


PERSON_SYNONYMS = {
    "person", "people", "human", "humans", "man", "men", "woman", "women",
    "boy", "boys", "girl", "girls", "child", "children", "kid", "kids",
    "toddler", "toddlers", "baby", "babies", "infant", "infants",
    "youth", "player", "players", "pedestrian", "pedestrians"
}


def is_person_class(cls_name: str) -> bool:
    """Check if class name corresponds to a person entity (adult, child, kid, player, etc.)."""
    if not cls_name:
        return False
    clean = cls_name.strip().lower()
    norm = normalize_class_name(clean)
    return norm == "person" or clean in PERSON_SYNONYMS


def is_main_object(cls_name: str, has_movement_or_interaction: bool = False) -> bool:
    """
    Check if an object category is a main, relevant visual object.
    Filters out unnecessary static background elements unless they actively undergo movement or interaction.
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
    Advanced Multi-Object Tracker with 1-to-1 Hungarian Data Association (`linear_sum_assignment`),
    Multi-Signal Cost Formulation (IoU, Center Distance, BBox Scale Ratio, Constant-Velocity Motion, Temporal Gap),
    and Explicit Track Lifecycle Management (TENTATIVE -> CONFIRMED -> LOST -> REMOVED).
    """

    def __init__(
        self,
        min_confirmed_hits: Optional[int] = None,
        max_lost_frames: Optional[int] = None,
        max_lost_seconds: Optional[float] = None,
        iou_threshold: float = 0.15,
    ):
        self.min_confirmed_hits = min_confirmed_hits if min_confirmed_hits is not None else getattr(settings, "TRACK_TENTATIVE_HITS", 2)
        self.max_lost_frames = max_lost_frames if max_lost_frames is not None else getattr(settings, "TRACK_MAX_LOST_FRAMES", 15)
        self.max_lost_seconds = max_lost_seconds if max_lost_seconds is not None else getattr(settings, "TRACK_MAX_LOST_SECONDS", 3.5)
        self.iou_threshold = iou_threshold

    def predict_track_bbox(self, track: TrackedObject, target_timestamp: float) -> List[float]:
        """Predict expected bounding box location using constant velocity model."""
        if not track.positions:
            return [0.0, 0.0, 0.0, 0.0]

        last_pos = track.positions[-1]["bbox"]
        if len(track.positions) < 2 or (track.velocity_x == 0.0 and track.velocity_y == 0.0):
            return list(last_pos)

        dt = max(0.0, target_timestamp - track.last_seen)
        dx = track.velocity_x * dt
        dy = track.velocity_y * dt

        return [
            round(last_pos[0] + dx, 1),
            round(last_pos[1] + dy, 1),
            round(last_pos[2] + dx, 1),
            round(last_pos[3] + dy, 1),
        ]

    def update_track_velocity(self, track: TrackedObject) -> None:
        """Update track velocity (vx, vy) in pixels/sec using last position entries."""
        if len(track.positions) < 2:
            track.velocity_x = 0.0
            track.velocity_y = 0.0
            return

        p1 = track.positions[-2]
        p2 = track.positions[-1]
        t1 = p1.get("timestamp", track.first_seen)
        t2 = p2.get("timestamp", track.last_seen)
        dt = max(0.01, t2 - t1)

        b1, b2 = p1["bbox"], p2["bbox"]
        cx1, cy1 = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
        cx2, cy2 = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0

        track.velocity_x = round((cx2 - cx1) / dt, 2)
        track.velocity_y = round((cy2 - cy1) / dt, 2)

    def calculate_association_cost(
        self,
        track: TrackedObject,
        det: YOLODetection,
        frame_timestamp: float
    ) -> float:
        """
        Calculate multi-signal assignment cost between track and detection.
        Combines (1 - IoU), normalized center distance, bbox scale ratio, velocity prediction, and gap penalty.
        Returns float cost between 0.0 (perfect match) and 1000.0 (invalid assignment).
        """
        track_norm_cls = normalize_class_name(track.canonical_name or track.object_type)
        det_norm_cls = normalize_class_name(det.class_name)

        # Class mismatch check
        if track_norm_cls != det_norm_cls:
            return 1000.0

        # Cannot assign to a track already updated in the current frame
        gap_sec = frame_timestamp - track.last_seen
        if gap_sec <= 0.01 or gap_sec > self.max_lost_seconds:
            return 1000.0

        # Predict expected position using motion model
        pred_bbox = self.predict_track_bbox(track, frame_timestamp)
        iou = calculate_bbox_iou(det.bbox, pred_bbox)
        raw_iou = calculate_bbox_iou(det.bbox, track.positions[-1]["bbox"]) if track.positions else iou

        dist = calculate_center_distance(det.bbox, pred_bbox)
        raw_dist = calculate_center_distance(det.bbox, track.positions[-1]["bbox"]) if track.positions else dist

        # Use velocity model predicted position when moving, fallback to raw last position if stationary
        has_velocity = len(track.positions) >= 2 and (abs(track.velocity_x) > 5.0 or abs(track.velocity_y) > 5.0)
        effective_iou = iou if has_velocity else max(iou, raw_iou)
        effective_dist = dist if has_velocity else min(dist, raw_dist)

        # Bbox dimension & scale consistency check
        last_bbox = track.positions[-1]["bbox"] if track.positions else det.bbox
        last_w = max(1.0, last_bbox[2] - last_bbox[0])
        last_h = max(1.0, last_bbox[3] - last_bbox[1])
        det_w = max(1.0, det.bbox[2] - det.bbox[0])
        det_h = max(1.0, det.bbox[3] - det.bbox[1])

        scale_ratio = max(det_w / last_w, last_w / det_w, det_h / last_h, last_h / det_h)
        if scale_ratio > 3.5:  # Sudden size jump is impossible for physical objects
            return 1000.0

        box_scale = max(last_w, last_h, 40.0)
        norm_dist = effective_dist / box_scale

        # Multi-signal Cost formulation
        w_iou = 0.40
        w_dist = 0.35
        w_scale = 0.15
        w_gap = 0.10

        iou_cost = 1.0 - effective_iou
        dist_cost = min(1.0, norm_dist / 2.0)
        scale_cost = min(1.0, (scale_ratio - 1.0) / 2.0)
        gap_cost = min(1.0, gap_sec / self.max_lost_seconds)

        if effective_iou >= 0.05:
            cost = w_iou * iou_cost + w_dist * dist_cost + w_scale * scale_cost + w_gap * gap_cost
        elif effective_dist <= box_scale * 3.0 or effective_dist <= getattr(settings, "TRACK_MERGE_MAX_DISTANCE_PX", 180.0):
            # Lower overlap due to rapid movement -> rely on center distance & size consistency
            cost = 0.35 + 0.40 * dist_cost + 0.15 * scale_cost + 0.10 * gap_cost
        else:
            cost = 1000.0

        return float(cost)

    def track_entities(
        self,
        sampled_frames: List[SampledFrame],
        frame_detections: Dict[str, List[YOLODetection]],
        frame_observations: List[FrameObservation] = None
    ) -> List[TrackedObject]:
        """
        Build persistent 1-to-1 TrackedObject trajectories using Hungarian matching and explicit lifecycles.
        """
        active_tracks: Dict[str, TrackedObject] = {}
        class_counters: Dict[str, int] = {}
        obs_by_frame = {obs.frame_id: obs for obs in (frame_observations or [])}

        sorted_frames = sorted(sampled_frames, key=lambda f: f.timestamp)

        for frame in sorted_frames:
            raw_detections = frame_detections.get(frame.frame_id, [])

            # Intra-frame NMS deduplication for person detections
            person_dets = [d for d in raw_detections if normalize_class_name(d.class_name) == "person"]
            non_person_dets = [d for d in raw_detections if normalize_class_name(d.class_name) != "person"]

            deduped_person_dets = []
            for p_det in sorted(person_dets, key=lambda d: d.confidence, reverse=True):
                is_dup = False
                for kept in deduped_person_dets:
                    iou = calculate_bbox_iou(p_det.bbox, kept.bbox)
                    dist = calculate_center_distance(p_det.bbox, kept.bbox)
                    if iou >= getattr(settings, "PERSON_NMS_IOU", 0.40) or (dist < 20.0 and iou > 0.25):
                        is_dup = True
                        break
                if not is_dup:
                    deduped_person_dets.append(p_det)

            detections = deduped_person_dets + non_person_dets
            vlm_obs = obs_by_frame.get(frame.frame_id)

            # Filter active tracks (TENTATIVE, CONFIRMED, LOST)
            track_keys = [
                tid for tid, trk in active_tracks.items()
                if trk.track_state in ("TENTATIVE", "CONFIRMED", "LOST")
            ]
            tracks_list = [active_tracks[tid] for tid in track_keys]

            matched_det_indices = set()
            matched_track_keys = set()

            if tracks_list and detections:
                cost_matrix = np.full((len(tracks_list), len(detections)), 1000.0)

                for t_idx, trk in enumerate(tracks_list):
                    for d_idx, det in enumerate(detections):
                        cost_matrix[t_idx, d_idx] = self.calculate_association_cost(trk, det, frame.timestamp)

                row_ind, col_ind = solve_linear_sum_assignment(cost_matrix)

                for r, c in zip(row_ind, col_ind):
                    c_val = cost_matrix[r, c]
                    if c_val < 0.75:  # Valid assignment cost threshold
                        matched_track_keys.add(track_keys[r])
                        matched_det_indices.add(c)

                        track = tracks_list[r]
                        det = detections[c]

                        # Update matched track
                        det.track_id = track.track_id
                        track.last_seen = frame.timestamp
                        track.detection_count += 1
                        track.hits_count += 1
                        track.lost_frames_count = 0

                        # Lifecycle transition check
                        if track.hits_count >= self.min_confirmed_hits:
                            track.track_state = "CONFIRMED"
                        else:
                            track.track_state = "TENTATIVE"

                        track.avg_confidence = round(
                            (track.avg_confidence * (track.detection_count - 1) + det.confidence) / track.detection_count,
                            3
                        )
                        track.positions.append({"timestamp": frame.timestamp, "bbox": det.bbox})
                        track.state_history.append({"timestamp": frame.timestamp, "state": "tracked_position"})
                        self.update_track_velocity(track)

            # Process unmatched active tracks -> Transition to LOST or REMOVED
            for tid in track_keys:
                if tid not in matched_track_keys:
                    track = active_tracks[tid]
                    track.lost_frames_count += 1
                    gap_sec = frame.timestamp - track.last_seen

                    if track.lost_frames_count > self.max_lost_frames or gap_sec > self.max_lost_seconds:
                        track.track_state = "REMOVED"
                        track.state_history.append({"timestamp": frame.timestamp, "state": "removed"})
                    else:
                        track.track_state = "LOST"
                        track.state_history.append({"timestamp": frame.timestamp, "state": "lost"})

            # Process unmatched detections -> Try trajectory re-association before spawning new track
            for d_idx, det in enumerate(detections):
                if d_idx not in matched_det_indices:
                    raw_cls = det.class_name
                    norm_cls = normalize_class_name(raw_cls)
                    display_cls = "Ball" if norm_cls == "sports ball" else norm_cls.capitalize()

                    # Check trajectory reassociation with existing recent tracks of same class
                    # CRITICAL FIX: gap_sec must be > 0.05 to prevent merging distinct entities co-existing in the SAME frame!
                    reassociated_track = None
                    max_merge_dist = getattr(settings, "TRACK_MERGE_MAX_DISTANCE_PX", 180.0) if norm_cls == "person" else 220.0
                    max_merge_gap = getattr(settings, "TRACK_MERGE_MAX_GAP_SEC", 5.0)

                    for existing_trk in active_tracks.values():
                        if normalize_class_name(existing_trk.canonical_name or existing_trk.object_type) == norm_cls:
                            gap_sec = frame.timestamp - existing_trk.last_seen
                            if 0.05 < gap_sec <= max_merge_gap:
                                last_bbox = existing_trk.positions[-1]["bbox"] if existing_trk.positions else [0, 0, 0, 0]
                                dist = calculate_center_distance(det.bbox, last_bbox)
                                pred_bbox = self.predict_track_bbox(existing_trk, frame.timestamp)
                                pred_dist = calculate_center_distance(det.bbox, pred_bbox)

                                if min(dist, pred_dist) <= max_merge_dist:
                                    reassociated_track = existing_trk
                                    break

                    if reassociated_track is not None:
                        # Merge unmatched detection into existing track
                        det.track_id = reassociated_track.track_id
                        reassociated_track.last_seen = frame.timestamp
                        reassociated_track.detection_count += 1
                        reassociated_track.hits_count += 1
                        reassociated_track.lost_frames_count = 0
                        reassociated_track.track_state = "CONFIRMED"
                        reassociated_track.positions.append({"timestamp": frame.timestamp, "bbox": det.bbox})
                        reassociated_track.state_history.append({"timestamp": frame.timestamp, "state": "reassociated"})
                        self.update_track_velocity(reassociated_track)
                        logger.info(f"Re-associated unmatched person detection into existing track {reassociated_track.track_id}")
                    else:
                        # Spawn new TENTATIVE track
                        if display_cls not in class_counters:
                            class_counters[display_cls] = 1

                        new_track_id = f"{display_cls} #{class_counters[display_cls]}"
                        class_counters[display_cls] += 1
                        det.track_id = new_track_id

                        initial_state = "CONFIRMED" if self.min_confirmed_hits <= 1 else "TENTATIVE"

                        active_tracks[new_track_id] = TrackedObject(
                            track_id=new_track_id,
                            object_type=raw_cls,
                            canonical_name=norm_cls,
                            first_seen=frame.timestamp,
                            last_seen=frame.timestamp,
                            positions=[{"timestamp": frame.timestamp, "bbox": det.bbox}],
                            activities=[],
                            interactions=[],
                            lifecycle_events=["appeared"],
                            state_history=[{"timestamp": frame.timestamp, "state": "appeared"}],
                            detection_count=1,
                            avg_confidence=det.confidence,
                            track_state=initial_state,
                            hits_count=1,
                            lost_frames_count=0,
                            is_unique_person=(norm_cls == "person"),
                            is_unique_object=(norm_cls != "person"),
                        )

            # Attach VLM activities & interactions
            if vlm_obs:
                for det in detections:
                    if det.track_id and det.track_id in active_tracks:
                        trk = active_tracks[det.track_id]
                        for act in vlm_obs.activities:
                            if act not in trk.activities:
                                trk.activities.append(act)
                        for inter in vlm_obs.interactions:
                            if inter not in trk.interactions:
                                trk.interactions.append(inter)

        # Filter confirmed/valid tracks for output
        output_tracks: List[TrackedObject] = []
        num_frames = len(sorted_frames)

        for trk in active_tracks.values():
            is_person = is_person_class(trk.canonical_name) or is_person_class(trk.object_type)
            
            # Enforce confirmation hit requirements
            if is_person:
                # Require min_confirmed_hits unless total sampled frames is 1
                min_req = 1 if num_frames <= 1 else self.min_confirmed_hits
                is_valid = trk.hits_count >= min_req or trk.track_state == "CONFIRMED"
            else:
                is_valid = trk.hits_count >= 1 or trk.track_state == "CONFIRMED"

            if is_valid:
                trk.track_state = "CONFIRMED"
                trk.is_unique_person = is_person
                trk.is_unique_object = not is_person
                output_tracks.append(trk)

        # Calculate movement vector & lifecycle events
        for trk in output_tracks:
            if len(trk.positions) > 1:
                p_first = trk.positions[0]["bbox"]
                p_last = trk.positions[-1]["bbox"]
                cx1, cy1 = (p_first[0] + p_first[2]) / 2.0, (p_first[1] + p_first[3]) / 2.0
                cx2, cy2 = (p_last[0] + p_last[2]) / 2.0, (p_last[1] + p_last[3]) / 2.0

                dist = float(np.hypot(cx2 - cx1, cy2 - cy1))
                trk.movement_distance = round(dist, 1)

                if dist > 40.0:
                    trk.movement_confidence = min(0.98, round(0.50 + dist / 300.0, 2))
                    if "moved" not in trk.lifecycle_events:
                        trk.lifecycle_events.append("moved")
                        trk.state_history.append({"timestamp": trk.last_seen, "state": "moved"})

                    # Direction vector calculation
                    dx, dy = cx2 - cx1, cy2 - cy1
                    if abs(dx) > abs(dy):
                        trk.movement_direction = "RIGHT" if dx > 0 else "LEFT"
                    else:
                        trk.movement_direction = "DOWN" if dy > 0 else "UP"

            if sorted_frames and (sorted_frames[-1].timestamp - trk.last_seen) > 4.0:
                if "exited" not in trk.lifecycle_events:
                    trk.lifecycle_events.append("exited")
                    trk.state_history.append({"timestamp": trk.last_seen, "state": "exited"})

        people_cnt = len([t for t in output_tracks if t.canonical_name == "person"])
        obj_cnt = len([t for t in output_tracks if t.canonical_name != "person"])
        logger.info(
            f"Hungarian IoU Tracker built trajectories for {len(output_tracks)} confirmed entities "
            f"({people_cnt} unique people, {obj_cnt} unique objects)."
        )

        return output_tracks


object_tracker = SpatialIoUTracker()

