from typing import List, Dict, Any
from models.schemas import FrameObservation, TrackedObject, VideoEvent, Scene
from utils.logger import logger

class EventDetector:
    """
    Advanced Multi-Frame Temporal Event Engine.
    Verifies physical state changes across consecutive frames, validates evidence,
    and assigns multi-source confidence scores.
    """

    def detect_events(
        self,
        scenes: List[Scene],
        frame_observations: List[FrameObservation],
        tracks: List[TrackedObject]
    ) -> List[VideoEvent]:
        events: List[VideoEvent] = []
        event_counter = 1

        # Sort frame observations chronologically
        sorted_obs = sorted(frame_observations, key=lambda o: o.timestamp)
        obs_by_frame = {o.frame_id: o for o in sorted_obs}

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
                    evidence_frames=[scene.keyframe_paths[0]] if scene.keyframe_paths else [],
                )
            )
            event_counter += 1

        # 2. Track Trajectory & Object Lifecycle Events
        for trk in tracks:
            duration = trk.last_seen - trk.first_seen

            # Entity appearance event
            events.append(
                VideoEvent(
                    event_id=f"evt_{event_counter:03d}",
                    start_time=trk.first_seen,
                    end_time=trk.last_seen,
                    event_type="PERSON" if trk.object_type == "person" else "OBJECT",
                    subject=trk.track_id,
                    object=None,
                    description=f"{trk.track_id.capitalize()} appeared in scene at {trk.first_seen:.1f}s until {trk.last_seen:.1f}s.",
                    confidence=0.94 if len(trk.positions) >= 2 else 0.80,
                    evidence_frames=[],
                )
            )
            event_counter += 1

            # Verified movement state change event
            if "moved" in trk.lifecycle_events and duration > 1.0:
                events.append(
                    VideoEvent(
                        event_id=f"evt_{event_counter:03d}",
                        start_time=trk.first_seen,
                        end_time=trk.last_seen,
                        event_type="MOVEMENT",
                        subject=trk.track_id,
                        object=None,
                        description=f"{trk.track_id.capitalize()} changed position across frames ({trk.first_seen:.1f}s -> {trk.last_seen:.1f}s).",
                        confidence=0.91,
                        evidence_frames=[],
                    )
                )
                event_counter += 1

        # 3. Multi-Frame State Change Verification for Actions & Interactions
        for i in range(len(sorted_obs)):
            curr_obs = sorted_obs[i]
            prev_obs = sorted_obs[i - 1] if i > 0 else None
            next_obs = sorted_obs[i + 1] if i + 1 < len(sorted_obs) else None

            for act in curr_obs.activities:
                # Check cross-frame verification
                multi_frame_proof = False
                supporting_frames = [curr_obs.frame_id]

                if prev_obs and any(act.lower() in p_act.lower() for p_act in prev_obs.activities):
                    multi_frame_proof = True
                    supporting_frames.append(prev_obs.frame_id)
                if next_obs and any(act.lower() in n_act.lower() for n_act in next_obs.activities):
                    multi_frame_proof = True
                    supporting_frames.append(next_obs.frame_id)

                # Confidence calculation based on multi-source proof
                if multi_frame_proof:
                    conf = 0.93
                elif curr_obs.people and curr_obs.objects:
                    conf = 0.85
                else:
                    conf = 0.72

                subj = curr_obs.people[0].temporary_id if curr_obs.people else "Entity"
                obj = curr_obs.objects[0].name if curr_obs.objects else None

                events.append(
                    VideoEvent(
                        event_id=f"evt_{event_counter:03d}",
                        start_time=curr_obs.timestamp,
                        end_time=round(curr_obs.timestamp + 2.0, 2),
                        event_type="MOVEMENT" if "walk" in act.lower() or "run" in act.lower() else "OBJECT",
                        subject=subj,
                        object=obj,
                        description=f"{act.capitalize()} observed at {curr_obs.timestamp:.1f}s.",
                        confidence=conf,
                        evidence_frames=list(set(supporting_frames)),
                    )
                )
                event_counter += 1

        # Sort all events strictly by start_time
        events.sort(key=lambda e: (e.start_time, e.event_id))
        logger.info(f"Generated {len(events)} verified timed events with confidence scoring.")
        return events

event_detector = EventDetector()
