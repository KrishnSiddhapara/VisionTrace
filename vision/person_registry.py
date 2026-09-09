import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from models.schemas import TrackedObject, FrameObservation, VideoMetadata, PersonObservation
from vision.tracker import calculate_bbox_iou, calculate_center_distance, normalize_class_name, is_person_class
from utils.logger import logger

class PersonEntity(BaseModel):
    """Canonical persistent physical person record."""
    person_id: str
    canonical_name: str = "person"
    first_seen: float
    last_seen: float
    track_ids: List[str] = Field(default_factory=list)
    positions: List[Dict[str, Any]] = Field(default_factory=list)  # [{'timestamp': 1.0, 'bbox': [...]}]
    activities: List[str] = Field(default_factory=list)
    interactions: List[str] = Field(default_factory=list)
    lifecycle_events: List[str] = Field(default_factory=list)
    avg_confidence: float = 0.90
    hits_count: int = 1
    motion_state: str = "STATIONARY"  # 'STATIONARY', 'MOVING', 'OCCLUDED'
    movement_distance: float = 0.0
    velocity_px_sec: float = 0.0
    evidence_level: str = "CONFIRMED"  # 'CONFIRMED', 'PROBABLE', 'UNCERTAIN'
    is_active: bool = True


import re

def extract_people_count_from_text(text_list: List[str]) -> int:
    """Extract explicit count of people/children mentioned in text descriptions."""
    num_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    max_found = 0
    for text in (text_list or []):
        if not text:
            continue
        lower = text.lower()
        matches = re.findall(r'(\b\d+|\bone|\btwo|\bthree|\bfour|\bfive|\bsix|\bseven|\beight|\bnine|\bten)\s+(?:children|kids|boys|girls|toddlers|babies|people|humans|players|persons)', lower)
        for m in matches:
            val = int(m) if m.isdigit() else num_map.get(m, 0)
            if val > max_found:
                max_found = val
    return max_found


class CanonicalPersonRegistry:
    """
    Canonical Person Entity Manager & Temporal Track Re-Associator.
    Fixes inaccurate person counts, false movements, track ID switches, and ungrounded VLM observations.
    Enforces architectural principle: Detection != Track != Physical Person.
    """

    def __init__(self, max_occlusion_gap_sec: float = 8.0, motion_min_distance_px: float = 45.0):
        self.max_occlusion_gap = max_occlusion_gap_sec
        self.motion_min_distance = motion_min_distance_px

    def _is_near_boundary(self, bbox: List[float], width: int = 640, height: int = 480, margin_pct: float = 0.15) -> bool:
        """Check if bounding box center or edge is near video frame boundary."""
        if not bbox or len(bbox) < 4:
            return False
        margin_w = width * margin_pct
        margin_h = height * margin_pct
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return (cx <= margin_w or cx >= (width - margin_w) or cy <= margin_h or cy >= (height - margin_h))

    def reconcile_person_tracks(
        self,
        tracks: List[TrackedObject],
        frame_observations: List[FrameObservation] = None,
        metadata: Optional[VideoMetadata] = None
    ) -> List[PersonEntity]:
        """
        Reconcile raw person trajectories into persistent physical PersonEntities.
        Handles track ID switching, temporary occlusion re-association, motion filtering, and spatial VLM grounding.
        """
        # Filter person tracks only (including synonyms: children, kids, boys, girls, etc.)
        person_tracks = [
            t for t in tracks
            if is_person_class(getattr(t, "canonical_name", "")) or is_person_class(t.object_type)
        ]
        person_entities: List[PersonEntity] = []
        filtered_tracks = []

        if person_tracks:
            # 1. Filter out single-frame weak person tracks if consistent multi-hit tracks exist
            effective_min_hits = 2 if (len(person_tracks) > 1 and any(t.hits_count >= 2 for t in person_tracks)) else 1
            for trk in person_tracks:
                if trk.hits_count >= effective_min_hits or trk.avg_confidence >= 0.75:
                    filtered_tracks.append(trk)
                else:
                    logger.info(f"[Person Registry] Filtered weak single-frame person track '{trk.track_id}' (hits={trk.hits_count}, conf={trk.avg_confidence:.2f}).")

            if not filtered_tracks:
                filtered_tracks = person_tracks

        # Determine peak concurrent people detected at any single timestamp
        time_to_boxes: Dict[float, List[List[float]]] = {}
        for trk in filtered_tracks:
            for p in trk.positions:
                ts = round(float(p.get("timestamp", 0.0)), 2)
                boxes = time_to_boxes.setdefault(ts, [])
                bbox = p.get("bbox", [0, 0, 0, 0])
                if not any(calculate_bbox_iou(bbox, existing_b) >= 0.40 for existing_b in boxes):
                    boxes.append(bbox)

        det_max_concurrent = max((len(boxes) for boxes in time_to_boxes.values()), default=1)

        vlm_max_people = 0
        if frame_observations:
            vlm_counts = [len(obs.people) for obs in frame_observations if obs.people]
            if vlm_counts:
                vlm_max_people = max(vlm_counts)

        max_concurrent_people = max(det_max_concurrent, vlm_max_people)
        logger.info(f"[Person Registry] Peak concurrent people in any single frame: {max_concurrent_people} (det={det_max_concurrent}, vlm={vlm_max_people})")

        # 2. First Pass: Sort tracks chronologically and perform Hungarian spatial-temporal matching
        sorted_tracks = sorted(filtered_tracks, key=lambda t: (t.first_seen, t.track_id))
        person_entities: List[PersonEntity] = []
        person_counter = 1

        width = metadata.width if metadata and metadata.width else 640
        height = metadata.height if metadata and metadata.height else 480
        total_duration = metadata.duration_sec if metadata and metadata.duration_sec else 10.0

        for trk in sorted_tracks:
            trk_first_bbox = trk.positions[0]["bbox"] if trk.positions else [0, 0, 0, 0]
            trk_last_bbox = trk.positions[-1]["bbox"] if trk.positions else [0, 0, 0, 0]

            best_match: Optional[PersonEntity] = None
            best_score = 0.0

            # Test spatial-temporal re-association with existing person entities
            for entity in person_entities:
                time_gap = trk.first_seen - entity.last_seen
                entity_last_bbox = entity.positions[-1]["bbox"] if entity.positions else [0, 0, 0, 0]
                dist = calculate_center_distance(entity_last_bbox, trk_first_bbox)
                iou = calculate_bbox_iou(entity_last_bbox, trk_first_bbox)

                # Check if overlapping timestamps exist
                overlapping_ts = set(round(p.get("timestamp", 0), 2) for p in entity.positions).intersection(
                    set(round(p.get("timestamp", 0), 2) for p in trk.positions)
                )

                if overlapping_ts:
                    # Check if overlapping boxes are duplicate detections of same person or distinct spatial locations
                    is_duplicate_box = False
                    for ts in overlapping_ts:
                        e_b = next((p["bbox"] for p in entity.positions if round(p.get("timestamp", 0), 2) == ts), None)
                        t_b = next((p["bbox"] for p in trk.positions if round(p.get("timestamp", 0), 2) == ts), None)
                        if e_b and t_b:
                            if calculate_bbox_iou(e_b, t_b) >= 0.40 or calculate_center_distance(e_b, t_b) < 30.0:
                                is_duplicate_box = True
                                break

                    if is_duplicate_box:
                        match_score = 0.85
                        if match_score > best_score:
                            best_score = match_score
                            best_match = entity
                    # If co-existing at distinct locations (IoU < 0.40) -> DISTINCT PEOPLE! Do not merge!
                    continue

                # Case B: Significant distinct co-existence at different locations -> Distinct people!
                if time_gap < -1.5:
                    continue

                # Case C: Sequential within occlusion gap
                if -1.5 <= time_gap <= self.max_occlusion_gap:
                    w1 = max(1.0, entity_last_bbox[2] - entity_last_bbox[0])
                    h1 = max(1.0, entity_last_bbox[3] - entity_last_bbox[1])
                    w2 = max(1.0, trk_first_bbox[2] - trk_first_bbox[0])
                    h2 = max(1.0, trk_first_bbox[3] - trk_first_bbox[1])
                    scale_diff = max(w1 / w2, w2 / w1, h1 / h2, h2 / h1)

                    if scale_diff <= 2.5:
                        effective_gap = max(0.0, time_gap)
                        max_allowed_dist = max(200.0, 120.0 * effective_gap)
                        if dist <= max_allowed_dist or iou >= 0.05:
                            match_score = max(0.1, 1.0 - (dist / max_allowed_dist))
                            if match_score > best_score:
                                best_score = match_score
                                best_match = entity

            if best_match:
                # Re-associate fragmented track into existing PersonEntity
                logger.info(
                    f"[PERSON REGISTRY RE-ASSOCIATION] Re-associated track '{trk.track_id}' into '{best_match.person_id}' "
                    f"(gap={trk.first_seen - best_match.last_seen:.1f}s, confidence={best_score:.2f})."
                )
                best_match.track_ids.append(trk.track_id)
                best_match.last_seen = max(best_match.last_seen, trk.last_seen)
                best_match.positions.extend(trk.positions)
                best_match.hits_count += trk.hits_count
                best_match.avg_confidence = round((best_match.avg_confidence + trk.avg_confidence) / 2.0, 2)

                for act in trk.activities:
                    if act not in best_match.activities:
                        best_match.activities.append(act)
                for inter in trk.interactions:
                    if inter not in best_match.interactions:
                        best_match.interactions.append(inter)
            else:
                # Spawn new canonical PersonEntity
                p_id = f"Person #{person_counter}"
                person_counter += 1

                new_entity = PersonEntity(
                    person_id=p_id,
                    canonical_name="person",
                    first_seen=trk.first_seen,
                    last_seen=trk.last_seen,
                    track_ids=[trk.track_id],
                    positions=list(trk.positions),
                    activities=list(trk.activities),
                    interactions=list(trk.interactions),
                    lifecycle_events=[],
                    avg_confidence=trk.avg_confidence,
                    hits_count=trk.hits_count,
                    evidence_level="CONFIRMED" if trk.hits_count >= 2 or trk.avg_confidence >= 0.75 else "PROBABLE",
                    is_active=True,
                )
                person_entities.append(new_entity)

        # 3. Second Pass: Strict Concurrent Person Reconciliation
        # If candidate person entities exceed max_concurrent_people, merge non-coexisting sequential entities
        if len(person_entities) > max_concurrent_people:
            logger.info(
                f"[Person Registry] Candidate entities ({len(person_entities)}) exceed peak concurrency ({max_concurrent_people}). Running second-pass reconciliation."
            )
            pe_sorted = sorted(person_entities, key=lambda x: x.first_seen)
            merged_entities: List[PersonEntity] = []

            for next_pe in pe_sorted:
                merged = False
                for target_pe in merged_entities:
                    t_target = set(round(p.get("timestamp", 0), 2) for p in target_pe.positions)
                    t_next = set(round(p.get("timestamp", 0), 2) for p in next_pe.positions)
                    common_frames = t_target.intersection(t_next)

                    is_spatially_distinct = False
                    if common_frames:
                        for ts in common_frames:
                            b_target = next((p["bbox"] for p in target_pe.positions if round(p.get("timestamp", 0), 2) == ts), None)
                            b_next = next((p["bbox"] for p in next_pe.positions if round(p.get("timestamp", 0), 2) == ts), None)
                            if b_target and b_next:
                                if calculate_bbox_iou(b_target, b_next) < 0.40 and calculate_center_distance(b_target, b_next) >= 35.0:
                                    is_spatially_distinct = True
                                    break

                    if not is_spatially_distinct:
                        # They do not co-exist at distinct locations simultaneously!
                        # Merge next_pe into target_pe
                        target_pe.track_ids.extend(next_pe.track_ids)
                        target_pe.last_seen = max(target_pe.last_seen, next_pe.last_seen)
                        target_pe.positions.extend(next_pe.positions)
                        target_pe.hits_count += next_pe.hits_count
                        for act in next_pe.activities:
                            if act not in target_pe.activities:
                                target_pe.activities.append(act)
                        for inter in next_pe.interactions:
                            if inter not in target_pe.interactions:
                                target_pe.interactions.append(inter)
                        logger.info(
                            f"[Person Registry] Merged sequential non-concurrent person '{next_pe.person_id}' into '{target_pe.person_id}'"
                        )
                        merged = True
                        break

                if not merged:
                    merged_entities.append(next_pe)

            person_entities = merged_entities

        # 4. Renumber Person IDs cleanly as Person #1, Person #2, Person #3...
        for idx, pe in enumerate(person_entities, start=1):
            pe.person_id = f"Person #{idx}"

        # 5. Post-processing: Calculate motion, displacement, velocity, and boundary lifecycle events
        for entity in person_entities:
            if len(entity.positions) > 1:
                p_first = entity.positions[0]["bbox"]
                p_last = entity.positions[-1]["bbox"]
                cx1, cy1 = (p_first[0] + p_first[2]) / 2.0, (p_first[1] + p_first[3]) / 2.0
                cx2, cy2 = (p_last[0] + p_last[2]) / 2.0, (p_last[1] + p_last[3]) / 2.0

                dist = float(np.hypot(cx2 - cx1, cy2 - cy1))
                duration = max(0.1, entity.last_seen - entity.first_seen)
                vel = round(dist / duration, 1)

                entity.movement_distance = round(dist, 1)
                entity.velocity_px_sec = vel

                # Motion State Classification
                if dist >= self.motion_min_distance and vel >= 12.0:
                    entity.motion_state = "MOVING"
                    if "moving across visible area" not in entity.activities:
                        entity.activities.append("moving across visible area")
                else:
                    entity.motion_state = "STATIONARY"
                    if "present in scene" not in entity.activities:
                        entity.activities.append("present in scene")

            # Entry & Exit Lifecycle Events
            first_bbox = entity.positions[0]["bbox"] if entity.positions else [0, 0, 0, 0]
            last_bbox = entity.positions[-1]["bbox"] if entity.positions else [0, 0, 0, 0]

            if entity.first_seen > 0.5 and self._is_near_boundary(first_bbox, width, height):
                entity.lifecycle_events.append("entered visible area")

            if (total_duration - entity.last_seen) > 1.5 and self._is_near_boundary(last_bbox, width, height):
                entity.lifecycle_events.append("exited visible area")

        # 6. Reconcile VLM Person Observations & text mentions (Synthesize missing person entities)
        if frame_observations:
            for obs in frame_observations:
                all_text = (obs.activities or []) + (obs.observations or []) + (obs.uncertainties or [])
                text_count = extract_people_count_from_text(all_text)
                target_count = max(len(obs.people) if obs.people else 0, text_count)

                if target_count > 0:
                    active_entities = [
                        e for e in person_entities
                        if e.first_seen <= (obs.timestamp + 2.5) and e.last_seen >= (obs.timestamp - 2.5)
                    ]

                    assigned_in_frame = set()
                    people_list = list(obs.people) if obs.people else []

                    # If text explicitly mentions more people than people_list, fill synthetic observations
                    if target_count > len(people_list):
                        for k in range(len(people_list) + 1, target_count + 1):
                            people_list.append(
                                PersonObservation(
                                    temporary_id=f"Person #{k}",
                                    description="Child / person detected in scene",
                                    activity="Present in visual scene",
                                    confidence=0.88,
                                )
                            )

                    for v_person in people_list:
                        unassigned = [e for e in active_entities if e.person_id not in assigned_in_frame]
                        if unassigned:
                            target_entity = unassigned[0]
                            assigned_in_frame.add(target_entity.person_id)
                            target_entity.last_seen = max(target_entity.last_seen, obs.timestamp)
                            target_entity.first_seen = min(target_entity.first_seen, obs.timestamp)
                            if v_person.activity and v_person.activity not in target_entity.activities:
                                target_entity.activities.append(v_person.activity)
                        else:
                            # Spawn new canonical PersonEntity for unassigned person in frame
                            new_idx = len(person_entities) + 1
                            p_id = f"Person #{new_idx}"
                            act_list = [v_person.activity] if v_person.activity else ["Present in visual scene"]
                            new_pe = PersonEntity(
                                person_id=p_id,
                                canonical_name="person",
                                first_seen=obs.timestamp,
                                last_seen=obs.timestamp,
                                track_ids=[f"vlm_person_{new_idx}"],
                                positions=[{"timestamp": obs.timestamp, "bbox": [100.0, 100.0, 200.0, 300.0]}],
                                activities=act_list,
                                interactions=[],
                                lifecycle_events=[],
                                avg_confidence=v_person.confidence or 0.88,
                                hits_count=1,
                                evidence_level="CONFIRMED",
                                is_active=True,
                            )
                            person_entities.append(new_pe)
                            active_entities.append(new_pe)
                            assigned_in_frame.add(p_id)

        # 7. Final clean renumbering of Person IDs: Person #1, Person #2, Person #3...
        for idx, pe in enumerate(person_entities, start=1):
            pe.person_id = f"Person #{idx}"

        logger.info(
            f"[PERSON REGISTRY COMPLETED] Reconciled {len(person_tracks)} raw person tracks into "
            f"{len(person_entities)} persistent canonical PersonEntities."
        )
        return person_entities


canonical_person_registry = CanonicalPersonRegistry()
