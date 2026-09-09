import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from models.schemas import TrackedObject, YOLODetection, FrameObservation
from vision.tracker import normalize_class_name, calculate_bbox_iou, calculate_center_distance
from utils.logger import logger

class PhysicalObject(BaseModel):
    object_id: str
    canonical_name: str
    object_type: str
    first_seen: float
    last_seen: float
    track_ids: List[str] = Field(default_factory=list)
    positions: List[Dict[str, Any]] = Field(default_factory=list)  # [{'timestamp': 1.0, 'bbox': [...]}]
    activities: List[str] = Field(default_factory=list)
    interactions: List[str] = Field(default_factory=list)
    lifecycle_events: List[str] = Field(default_factory=list)
    avg_confidence: float = 0.90
    is_active: bool = True

class PhysicalObjectRegistry:
    """
    Canonical Physical Object Registry & Temporal Reconciler.
    Enforces the fundamental architectural rule: Detection != Physical Object.
    
    Reconciles fragmented tracks across occlusions, fast motions (e.g. soccer ball passes),
    and bounding box loss into persistent physical object identities.
    """

    def __init__(self, max_reconciliation_gap: float = 8.0):
        self.max_gap_sec = max_reconciliation_gap

    def estimate_velocity(self, track: TrackedObject) -> Tuple[float, float]:
        """Estimate (vx, vy) center velocity in pixels per second from track position history."""
        if len(track.positions) < 2:
            return (0.0, 0.0)

        p1 = track.positions[-2]
        p2 = track.positions[-1]
        dt = max(0.01, p2["bbox"] if isinstance(p2, dict) and "timestamp" not in p2 else (p2.get("timestamp", 0) - p1.get("timestamp", 0)))
        if dt <= 0 or dt > 5.0:
            return (0.0, 0.0)

        b1 = p1["bbox"]
        b2 = p2["bbox"]
        c1_x, c1_y = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
        c2_x, c2_y = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0

        return ((c2_x - c1_x) / dt, (c2_y - c1_y) / dt)

    def reconcile_tracks(
        self,
        tracks: List[TrackedObject],
        frame_observations: List[FrameObservation] = None
    ) -> List[PhysicalObject]:
        """
        Reconcile a set of raw TrackedObject trajectories into unified PhysicalObjects.
        Solves multi-ball and duplicated object track fragmentation.
        """
        if not tracks:
            return []

        # Sort tracks chronologically by first_seen
        sorted_tracks = sorted(tracks, key=lambda t: t.first_seen)
        physical_objects: List[PhysicalObject] = []
        object_type_counters: Dict[str, int] = {}

        from vision.tracker import is_main_object

        for trk in sorted_tracks:
            norm_cls = normalize_class_name(trk.canonical_name or trk.object_type)
            has_movement = bool(trk.lifecycle_events or trk.activities or trk.interactions)
            if not is_main_object(norm_cls, has_movement_or_interaction=has_movement):
                continue

            trk_first_bbox = trk.positions[0]["bbox"] if trk.positions else [0, 0, 0, 0]
            trk_last_bbox = trk.positions[-1]["bbox"] if trk.positions else [0, 0, 0, 0]

            best_phys_match: Optional[PhysicalObject] = None
            best_score = 0.0

            # Search existing physical objects of identical category
            for phys in physical_objects:
                if phys.canonical_name == norm_cls:
                    time_gap = trk.first_seen - phys.last_seen

                    # Case A: Concurrent/Overlapping existence -> Physically distinct objects!
                    if time_gap < -0.5:
                        # Check if physically distinct in overlap frames
                        continue

                    # Case B: Sequential gap within threshold -> Test velocity prediction & spatial distance
                    if 0 <= time_gap <= self.max_gap_sec:
                        phys_last_bbox = phys.positions[-1]["bbox"] if phys.positions else [0, 0, 0, 0]
                        dist = calculate_center_distance(phys_last_bbox, trk_first_bbox)
                        iou = calculate_bbox_iou(phys_last_bbox, trk_first_bbox)

                        # For fast-moving small objects like sports balls:
                        if norm_cls == "sports ball":
                            # Allow larger spatial jump if time gap is reasonable (ball moving across field)
                            max_allowed_dist = max(250.0, 150.0 * time_gap)
                            if dist <= max_allowed_dist or iou > 0.05:
                                match_score = max(0.1, 1.0 - (dist / max_allowed_dist))
                                if match_score > best_score:
                                    best_score = match_score
                                    best_phys_match = phys
                        else:
                            # Standard objects (people, laptops, bags)
                            max_allowed_dist = max(180.0, 100.0 * time_gap)
                            if dist <= max_allowed_dist or iou >= 0.10:
                                match_score = max(0.1, 1.0 - (dist / max_allowed_dist))
                                if match_score > best_score:
                                    best_score = match_score
                                    best_phys_match = phys

            if best_phys_match:
                # Merge track into existing physical object
                best_phys_match.track_ids.append(trk.track_id)
                best_phys_match.last_seen = max(best_phys_match.last_seen, trk.last_seen)
                best_phys_match.positions.extend(trk.positions)
                
                # Merge activities & interactions
                for act in trk.activities:
                    if act not in best_phys_match.activities:
                        best_phys_match.activities.append(act)
                for inter in trk.interactions:
                    if inter not in best_phys_match.interactions:
                        best_phys_match.interactions.append(inter)
                for lc in trk.lifecycle_events:
                    if lc not in best_phys_match.lifecycle_events:
                        best_phys_match.lifecycle_events.append(lc)

                logger.info(
                    f"[Object Registry] Reconciled track '{trk.track_id}' into canonical physical object '{best_phys_match.object_id}'"
                )
            else:
                # Spawn new physical object
                display_type = "Ball" if norm_cls == "sports ball" else norm_cls.capitalize()
                object_type_counters[display_type] = object_type_counters.get(display_type, 0) + 1
                phys_id = f"{display_type} #{object_type_counters[display_type]}"

                new_phys = PhysicalObject(
                    object_id=phys_id,
                    canonical_name=norm_cls,
                    object_type=trk.object_type,
                    first_seen=trk.first_seen,
                    last_seen=trk.last_seen,
                    track_ids=[trk.track_id],
                    positions=list(trk.positions),
                    activities=list(trk.activities),
                    interactions=list(trk.interactions),
                    lifecycle_events=list(trk.lifecycle_events),
                    avg_confidence=trk.avg_confidence,
                    is_active=True
                )
                physical_objects.append(new_phys)

        logger.info(
            f"[Object Registry] Canonical reconciliation reduced {len(tracks)} tracks to {len(physical_objects)} physical objects."
        )

        # Post-reconciliation constraint: Merge non-concurrent tracks of the same category
        # Ensures that a single physical object (e.g. 1 soccer ball) isn't counted as multiple objects due to track breaks.
        by_category: Dict[str, List[PhysicalObject]] = {}
        for po in physical_objects:
            by_category.setdefault(po.canonical_name, []).append(po)

        final_reconciled: List[PhysicalObject] = []
        for cat, po_list in by_category.items():
            if len(po_list) <= 1:
                final_reconciled.extend(po_list)
                continue

            po_list_sorted = sorted(po_list, key=lambda x: x.first_seen)
            merged_list: List[PhysicalObject] = [po_list_sorted[0]]
            for next_po in po_list_sorted[1:]:
                prev_po = merged_list[-1]
                # Calculate time overlap between the two physical object tracks
                overlap = min(prev_po.last_seen, next_po.last_seen) - max(prev_po.first_seen, next_po.first_seen)
                # If non-overlapping or minimal overlap (e.g. < 1.5s), merge into single physical object identity
                if overlap <= 1.5:
                    prev_po.track_ids.extend(next_po.track_ids)
                    prev_po.last_seen = max(prev_po.last_seen, next_po.last_seen)
                    prev_po.positions.extend(next_po.positions)
                    for act in next_po.activities:
                        if act not in prev_po.activities:
                            prev_po.activities.append(act)
                    for inter in next_po.interactions:
                        if inter not in prev_po.interactions:
                            prev_po.interactions.append(inter)
                    for lc in next_po.lifecycle_events:
                        if lc not in prev_po.lifecycle_events:
                            prev_po.lifecycle_events.append(lc)
                else:
                    merged_list.append(next_po)
            final_reconciled.extend(merged_list)

        return final_reconciled


physical_object_registry = PhysicalObjectRegistry()

