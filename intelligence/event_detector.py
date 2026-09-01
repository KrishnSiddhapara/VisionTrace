from typing import List, Dict, Any, Tuple
from models.schemas import FrameObservation, TrackedObject, VideoEvent, Scene
from utils.logger import logger

class EventDetector:
    """
    Advanced Multi-Source Event Verification Pipeline.
    Strictly separates raw observations from confirmed events, calculates evidence-based confidence,
    assigns evidence levels (CONFIRMED/PROBABLE/UNCERTAIN/REJECTED), and performs semantic merging.
    """

    def calculate_event_confidence(
        self,
        supporting_frame_count: int,
        total_window_frames: int,
        has_tracking_support: bool,
        has_state_change: bool,
        vlm_evidence_strength: str,
        avg_quality_score: float,
    ) -> Tuple[float, str]:
        """
        Calculate evidence-grounded confidence score and evidence level.
        Returns (confidence, evidence_level).
        """
        # Base VLM strength score
        vlm_score = 0.90 if vlm_evidence_strength == "HIGH" else (0.75 if vlm_evidence_strength == "MEDIUM" else 0.60)
        
        # Frame support ratio
        frame_ratio = min(1.0, supporting_frame_count / max(1, total_window_frames))
        
        # Multi-source weighted combination
        score = (
            0.35 * vlm_score +
            0.30 * frame_ratio +
            0.20 * (1.0 if has_tracking_support else 0.5) +
            0.15 * (1.0 if has_state_change else 0.6)
        )
        score *= min(1.0, max(0.5, avg_quality_score))
        confidence = round(float(min(0.99, max(0.20, score))), 2)

        if confidence >= 0.85 and (supporting_frame_count >= 2 or has_tracking_support or has_state_change):
            level = "CONFIRMED"
        elif confidence >= 0.70:
            level = "PROBABLE"
        elif confidence >= 0.50:
            level = "UNCERTAIN"
        else:
            level = "REJECTED"

        return confidence, level

    def merge_consecutive_activities(
        self,
        sorted_obs: List[FrameObservation],
        tracks: List[TrackedObject]
    ) -> List[Dict[str, Any]]:
        """
        Group consecutive frame observations into candidate activity windows.
        Prevents emitting duplicate single-frame events.
        """
        candidate_groups: List[Dict[str, Any]] = []
        if not sorted_obs:
            return candidate_groups

        active_activities: Dict[str, Dict[str, Any]] = {}

        for obs in sorted_obs:
            if not obs.is_analyzed:
                continue

            current_acts = set(obs.activities + obs.interactions + obs.confirmed_changes)

            # Process active activities
            for act_text in list(active_activities.keys()):
                # Check if activity persists in current frame
                is_matching = any(
                    act_text.lower() in ca.lower() or ca.lower() in act_text.lower()
                    for ca in current_acts
                )
                if is_matching and (obs.timestamp - active_activities[act_text]["last_ts"]) <= 6.0:
                    # Extend activity window
                    active_activities[act_text]["last_ts"] = obs.timestamp
                    active_activities[act_text]["frames"].append(obs.frame_id)
                    active_activities[act_text]["quality_scores"].append(obs.quality_score)
                    active_activities[act_text]["vlm_strengths"].append(obs.evidence_strength)
                else:
                    # Close activity window
                    completed = active_activities.pop(act_text)
                    if len(completed["frames"]) >= 1:
                        candidate_groups.append(completed)

            # Start new activity windows
            for ca in current_acts:
                ca_norm = ca.strip()
                if ca_norm and ca_norm not in active_activities:
                    active_activities[ca_norm] = {
                        "activity": ca_norm,
                        "start_ts": obs.timestamp,
                        "last_ts": obs.timestamp,
                        "frames": [obs.frame_id],
                        "people": obs.people,
                        "objects": obs.objects,
                        "scene_id": obs.scene_id,
                        "quality_scores": [obs.quality_score],
                        "vlm_strengths": [obs.evidence_strength],
                    }

        # Flush remaining active activities
        for act_text, group in active_activities.items():
            candidate_groups.append(group)

        return candidate_groups

    def deduplicate_events(self, events: List[VideoEvent]) -> List[VideoEvent]:
        """
        Perform semantic event deduplication based on subject, object, time overlap, and description.
        """
        if not events:
            return []

        unique_events: List[VideoEvent] = []
        for evt in events:
            is_dup = False
            for existing in unique_events:
                # Check time overlap
                overlap = max(0.0, min(evt.end_time, existing.end_time) - max(evt.start_time, existing.start_time))
                same_type = evt.event_type == existing.event_type
                same_subject = evt.subject and existing.subject and (evt.subject.lower() in existing.subject.lower() or existing.subject.lower() in evt.subject.lower())
                same_desc = evt.description.lower() == existing.description.lower()

                if (overlap > 0.0 or same_desc) and same_type and same_subject:
                    is_dup = True
                    # Upgrade existing event with stronger confidence / merged frames
                    existing.end_time = max(existing.end_time, evt.end_time)
                    existing.confidence = max(existing.confidence, evt.confidence)
                    existing.evidence_frames = list(set(existing.evidence_frames + evt.evidence_frames))
                    if existing.confidence >= 0.85:
                        existing.evidence_level = "CONFIRMED"
                    break

            if not is_dup:
                unique_events.append(evt)

        return unique_events

    def detect_events(
        self,
        scenes: List[Scene],
        frame_observations: List[FrameObservation],
        tracks: List[TrackedObject]
    ) -> List[VideoEvent]:
        """
        Execute full Event Verification Pipeline.
        """
        events: List[VideoEvent] = []
        event_counter = 1

        sorted_obs = sorted(frame_observations, key=lambda o: o.timestamp)
        obs_by_frame = {o.frame_id: o for o in sorted_obs}
        track_ids = {trk.track_id for trk in tracks}

        # 1. Scene Boundary Events
        for scene in scenes:
            events.append(
                VideoEvent(
                    event_id=f"evt_{event_counter:03d}",
                    start_time=scene.start_time,
                    end_time=scene.end_time,
                    event_type="SCENE",
                    subject=f"Scene #{scene.scene_id}",
                    object=None,
                    description=f"Scene #{scene.scene_id} ({scene.start_time:.1f}s - {scene.end_time:.1f}s)",
                    confidence=1.0,
                    evidence_level="CONFIRMED",
                    evidence_frames=[scene.keyframe_paths[0]] if scene.keyframe_paths else [],
                    verification_status="VERIFIED",
                )
            )
            event_counter += 1

        # 2. Track Trajectory & Entity State Change Events
        for trk in tracks:
            duration = trk.last_seen - trk.first_seen
            if duration < 0.1:
                continue

            conf, level = self.calculate_event_confidence(
                supporting_frame_count=len(trk.positions),
                total_window_frames=max(1, len(sorted_obs)),
                has_tracking_support=True,
                has_state_change="moved" in trk.lifecycle_events or "exited" in trk.lifecycle_events,
                vlm_evidence_strength="HIGH",
                avg_quality_score=1.0,
            )

            if level != "REJECTED":
                events.append(
                    VideoEvent(
                        event_id=f"evt_{event_counter:03d}",
                        start_time=trk.first_seen,
                        end_time=trk.last_seen,
                        event_type="PERSON" if trk.object_type == "person" else "OBJECT",
                        subject=trk.track_id,
                        object=None,
                        description=f"{trk.track_id} observed from {trk.first_seen:.1f}s to {trk.last_seen:.1f}s.",
                        confidence=conf,
                        evidence_level=level,
                        evidence_frames=[],
                        verification_status="VERIFIED",
                    )
                )
                event_counter += 1

        # 3. Merged Candidate Activity Verification
        candidate_groups = self.merge_consecutive_activities(sorted_obs, tracks)

        for group in candidate_groups:
            act_text = group["activity"]
            start_ts = group["start_ts"]
            end_ts = group["last_ts"]
            frames = group["frames"]
            q_scores = group["quality_scores"]
            vlm_strengths = group["vlm_strengths"]

            avg_q = float(sum(q_scores) / max(1, len(q_scores)))
            top_vlm_strength = "HIGH" if "HIGH" in vlm_strengths else ("MEDIUM" if "MEDIUM" in vlm_strengths else "LOW")

            # Check tracking association
            has_tracking = any(
                p.temporary_id in track_ids for p in group["people"] if p.temporary_id
            ) or len(tracks) > 0

            # Check state change support
            has_state_change = any(
                kw in act_text.lower() for kw in ["pick", "drop", "move", "enter", "exit", "open", "close", "reach", "walk", "run"]
            )

            conf, level = self.calculate_event_confidence(
                supporting_frame_count=len(frames),
                total_window_frames=max(1, len(sorted_obs)),
                has_tracking_support=has_tracking,
                has_state_change=has_state_change,
                vlm_evidence_strength=top_vlm_strength,
                avg_quality_score=avg_q,
            )

            if level == "REJECTED":
                continue

            subj = group["people"][0].temporary_id if group["people"] else "Entity"
            obj = group["objects"][0].name if group["objects"] else None

            time_label = f"at {start_ts:.1f}s" if abs(end_ts - start_ts) < 0.5 else f"between {start_ts:.1f}s–{end_ts:.1f}s"
            desc = f"{act_text.capitalize()} observed {time_label}."

            events.append(
                VideoEvent(
                    event_id=f"evt_{event_counter:03d}",
                    start_time=start_ts,
                    end_time=max(end_ts, round(start_ts + 0.5, 2)),
                    event_type="MOVEMENT" if any(w in act_text.lower() for w in ["walk", "run", "move", "approach"]) else "OBJECT",
                    subject=subj,
                    object=obj,
                    description=desc,
                    confidence=conf,
                    evidence_level=level,
                    evidence_frames=frames,
                    verification_status="VERIFIED" if level in ("CONFIRMED", "PROBABLE") else "UNCERTAIN",
                )
            )
            event_counter += 1

        # 4. Semantic Deduplication
        deduped_events = self.deduplicate_events(events)
        deduped_events.sort(key=lambda e: (e.start_time, e.event_id))

        logger.info(f"Event Verification Pipeline produced {len(deduped_events)} verified timed events (from {len(events)} candidate events).")
        return deduped_events

event_detector = EventDetector()
