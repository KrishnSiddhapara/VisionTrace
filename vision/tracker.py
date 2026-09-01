import numpy as np
from typing import List, Dict, Any, Tuple
from models.schemas import SampledFrame, YOLODetection, TrackedObject, FrameObservation
from utils.logger import logger

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


class SpatialIoUTracker:
    """
    Tracks objects across sampled frames using Bounding Box Intersection-over-Union (IoU),
    gap-tolerant trajectory association, and state history transition modeling.
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
        Build persistent TrackedObject trajectories using BBox IoU overlap matching and state transitions.
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
                cls_name = det.class_name
                det_bbox = det.bbox

                best_match_id = None
                best_iou = 0.0

                # Search active tracks of same class within gap tolerance
                for track_id, track in active_tracks.items():
                    if track.object_type == cls_name and track_id not in matched_track_ids:
                        gap = frame.timestamp - track.last_seen
                        if gap <= self.max_gap_sec:
                            last_pos = track.positions[-1]["bbox"] if track.positions else None
                            if last_pos:
                                iou = calculate_bbox_iou(det_bbox, last_pos)
                                if iou >= self.iou_threshold and iou > best_iou:
                                    best_iou = iou
                                    best_match_id = track_id

                if best_match_id:
                    # Match found -> update existing track trajectory
                    track = active_tracks[best_match_id]
                    det.track_id = best_match_id
                    matched_track_ids.add(best_match_id)
                    track.last_seen = frame.timestamp
                    track.positions.append({"timestamp": frame.timestamp, "bbox": det_bbox})
                    track.state_history.append({"timestamp": frame.timestamp, "state": "tracked_position"})
                else:
                    # No match -> spawn new track with entity history
                    if cls_name not in class_counters:
                        class_counters[cls_name] = 1

                    new_track_id = f"{cls_name.capitalize()} #{class_counters[cls_name]}"
                    class_counters[cls_name] += 1
                    det.track_id = new_track_id
                    matched_track_ids.add(new_track_id)

                    active_tracks[new_track_id] = TrackedObject(
                        track_id=new_track_id,
                        object_type=cls_name,
                        first_seen=frame.timestamp,
                        last_seen=frame.timestamp,
                        positions=[{"timestamp": frame.timestamp, "bbox": det_bbox}],
                        activities=[],
                        interactions=[],
                        lifecycle_events=["appeared"],
                        state_history=[{"timestamp": frame.timestamp, "state": "appeared"}],
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
