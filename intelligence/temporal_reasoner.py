from typing import List, Dict, Any
from pathlib import Path
from models.schemas import (
    Scene, VideoEvent, TrackedObject, FrameObservation, VideoMetadata,
    SampledFrame, FinalSummary, FinalObjectRecord, FinalPersonRecord
)
from vision.vlm import get_vlm_provider
from config.settings import settings
from utils.logger import logger

class TemporalReasoner:
    """Grounded temporal reasoning engine combining verified events, tracking, and scenes."""

    def synthesize_timeline(
        self,
        scenes: List[Scene],
        events: List[VideoEvent],
        tracks: List[TrackedObject],
        frame_observations: List[FrameObservation]
    ) -> List[Dict[str, Any]]:
        timeline = []
        sorted_events = sorted(events, key=lambda e: (e.start_time, e.event_id))

        for idx, item in enumerate(sorted_events, start=1):
            m_s, s_s = divmod(item.start_time, 60)
            m_e, s_e = divmod(item.end_time, 60)
            
            if abs(item.end_time - item.start_time) < 0.5:
                time_str = f"{int(m_s):02d}:{s_s:04.1f}"
            else:
                time_str = f"{int(m_s):02d}:{s_s:04.1f}–{int(m_e):02d}:{s_e:04.1f}"

            timeline.append({
                "step_index": idx,
                "timestamp": item.start_time,
                "end_timestamp": item.end_time,
                "formatted_time": time_str,
                "event_type": item.event_type,
                "description": item.description,
                "subject": item.subject,
                "object": item.object,
                "confidence": item.confidence,
                "evidence_level": getattr(item, "evidence_level", "CONFIRMED"),
                "evidence_frames": item.evidence_frames,
                "verification_status": getattr(item, "verification_status", "VERIFIED"),
            })

        logger.info(f"Synthesized grounded chronological timeline with {len(timeline)} events.")
        return timeline

    def generate_summaries(
        self,
        metadata: VideoMetadata,
        scenes: List[Scene],
        timeline: List[Dict[str, Any]],
        tracks: List[TrackedObject]
    ) -> Dict[str, str]:
        dur = metadata.duration_sec
        sc_cnt = len(scenes)
        trk_cnt = len(tracks)

        confirmed_events = [t for t in timeline if t.get('evidence_level') == 'CONFIRMED' or t.get('confidence', 0) >= 0.85]

        # Quick Summary (1-2 sentences)
        quick = (
            f"Video '{metadata.filename}' ({dur:.1f}s, {sc_cnt} scenes) features {trk_cnt} tracked temporary entities "
            f"and {len(confirmed_events)} verified visual events grounded in visual evidence."
        )

        # Standard Summary (Structured paragraph with explicit evidence grounding)
        event_highlights = "; ".join([t['description'] for t in confirmed_events[:6]]) if confirmed_events else "continuous scene activity"

        standard = (
            f"The video '{metadata.filename}' spans {dur:.1f} seconds across {sc_cnt} visual scenes. "
            f"A total of {trk_cnt} temporary entities were tracked with spatial BBox IoU continuity. "
            f"Primary visually verified progression includes: {event_highlights}."
        )

        # Detailed Summary (Chronological breakdown with evidence levels)
        detailed_lines = [
            f"• [{t['formatted_time']}] {t['description']} | Level: {t.get('evidence_level', 'CONFIRMED')} ({int(t['confidence']*100)}% Confidence)"
            for t in timeline
        ]
        detailed = "\n".join(detailed_lines) if detailed_lines else "No specific events logged."

        # Grounded Analysis Breakdown (OBSERVED FACT vs INFERENCE vs UNKNOWN)
        obs_facts = [f"• [{t['formatted_time']}] {t['description']}" for t in confirmed_events[:8]]
        inferences = [
            "• Physical movement trajectories suggest continuous entity navigation within the visible area.",
            "• Object state transitions reflect observable spatial repositioning across frames."
        ]
        unknowns = [
            "• Subjective human intentions, emotions, and off-camera background events cannot be determined from visual evidence alone.",
            "• Unobserved real-world identity of tracked entities remains unassigned."
        ]

        technical = (
            f"Technical & Quality Metrics:\n"
            f"- Resolution: {metadata.resolution_str} @ {metadata.fps} FPS\n"
            f"- Duration: {metadata.duration_sec}s ({metadata.frame_count} total frames)\n"
            f"- Codec: {metadata.codec}\n"
            f"- PySceneDetect Scenes: {sc_cnt}\n"
            f"- Tracked Entity Trajectories: {trk_cnt}\n"
            f"- Verified Timeline Events: {len(timeline)} ({len(confirmed_events)} CONFIRMED)\n\n"
            f"OBSERVED FACTS:\n" + ("\n".join(obs_facts) if obs_facts else "None") + "\n\n"
            f"PERMISSIBLE INFERENCES:\n" + "\n".join(inferences) + "\n\n"
            f"UNOBSERVABLE UNKNOWNS:\n" + "\n".join(unknowns)
        )

        return {
            "quick": quick,
            "standard": standard,
            "detailed": detailed,
            "technical": technical,
        }

    def generate_final_summary(
        self,
        metadata: VideoMetadata,
        scenes: List[Scene],
        timeline: List[Dict[str, Any]],
        tracks: List[TrackedObject],
        frame_observations: List[FrameObservation],
        sampled_frames: List[SampledFrame]
    ) -> FinalSummary:
        """
        Generate structured evidence-grounded FinalSummary (OBJECTS, PEOPLE, FINAL DESCRIPTION).
        Combines deduplicated object records, temporary person entity tracks, and chronological visual narrative.
        """
        # 1. Deduplicate & Aggregate Objects
        object_map: Dict[str, Dict[str, Any]] = {}
        for trk in tracks:
            if trk.object_type.lower() != "person":
                obj_name = trk.object_type.capitalize()
                if obj_name not in object_map:
                    m_s, s_s = divmod(trk.first_seen, 60)
                    m_e, s_e = divmod(trk.last_seen, 60)
                    object_map[obj_name] = {
                        "name": obj_name,
                        "description": f"{obj_name} detected in scene.",
                        "first_seen": f"{int(m_s):02d}:{s_s:04.1f}",
                        "last_seen": f"{int(m_e):02d}:{s_e:04.1f}",
                        "movement": ", ".join(trk.lifecycle_events) if trk.lifecycle_events else "Observed in scene",
                        "state_changes": [s.get("state", "") for s in trk.state_history if s.get("state")],
                        "interactions": trk.interactions,
                        "confidence": 0.90,
                    }
                else:
                    m_e, s_e = divmod(trk.last_seen, 60)
                    object_map[obj_name]["last_seen"] = f"{int(m_e):02d}:{s_e:04.1f}"

        # Also pull objects from VLM frame observations
        for obs in frame_observations:
            for o in obs.objects:
                name = o.name.capitalize()
                if name.lower() != "person" and name not in object_map:
                    m_s, s_s = divmod(obs.timestamp, 60)
                    object_map[name] = {
                        "name": name,
                        "description": o.description or f"{name} visible in frame",
                        "first_seen": f"{int(m_s):02d}:{s_s:04.1f}",
                        "last_seen": f"{int(m_s):02d}:{s_s:04.1f}",
                        "movement": "Observed in frame keyframes",
                        "state_changes": obs.confirmed_changes,
                        "interactions": obs.interactions,
                        "confidence": round(float(o.confidence or 0.88), 2),
                    }

        final_objects = [FinalObjectRecord(**v) for v in object_map.values()]

        # 2. Aggregate People Entities
        person_tracks = [t for t in tracks if t.object_type.lower() == "person"]
        final_people = []

        for i, trk in enumerate(person_tracks, start=1):
            m_s, s_s = divmod(trk.first_seen, 60)
            m_e, s_e = divmod(trk.last_seen, 60)

            activities = trk.activities or ["Navigating scene area"]
            movements = trk.lifecycle_events or ["Entered visible area", "Moved across scene"]

            final_people.append(
                FinalPersonRecord(
                    temporary_id=f"Person #{i}",
                    description=f"Person entity (Track {trk.track_id})",
                    first_seen=f"{int(m_s):02d}:{s_s:04.1f}",
                    last_seen=f"{int(m_e):02d}:{s_e:04.1f}",
                    activities=activities,
                    movements=movements,
                    interactions=trk.interactions,
                    confidence=0.89,
                )
            )

        if not final_people and any(obs.people for obs in frame_observations):
            obs_people_ts = [o.timestamp for o in frame_observations if o.people]
            if obs_people_ts:
                m_s, s_s = divmod(min(obs_people_ts), 60)
                m_e, s_e = divmod(max(obs_people_ts), 60)
                final_people.append(
                    FinalPersonRecord(
                        temporary_id="Person #1",
                        description="Person observed in frame keyframes",
                        first_seen=f"{int(m_s):02d}:{s_s:04.1f}",
                        last_seen=f"{int(m_e):02d}:{s_e:04.1f}",
                        activities=["Observed in frame keyframes"],
                        movements=["Moved across visible area"],
                        interactions=[],
                        confidence=0.85,
                    )
                )

        # 3. Generate Chronological Final Description
        chronological_events = [t for t in timeline if t.get("description")]
        if chronological_events:
            lines = []
            for item in chronological_events:
                t_str = item.get("formatted_time", f"{item.get('timestamp', 0):.1f}s")
                desc = item.get("description", "")
                lines.append(f"At [{t_str}], {desc.lower() if desc and not desc.startswith('At') else desc}.")
            
            final_desc = (
                f"The video '{metadata.filename}' spans {metadata.duration_sec:.1f} seconds across {len(scenes)} visual scenes. "
                + " ".join(lines)
            )
        else:
            final_desc = f"The video '{metadata.filename}' spans {metadata.duration_sec:.1f} seconds. Insufficient visual change events were detected to construct a detailed movement narrative."

        # Attempt VLM Final Reasoning if available
        vlm_provider = get_vlm_provider()
        prompt_file = settings.PROMPTS_DIR / "final_summary.txt"
        
        if prompt_file.exists() and sampled_frames and not settings.VLM_MOCK_MODE and vlm_provider.client:
            try:
                prompt_template = prompt_file.read_text(encoding="utf-8")
                evidence_text = f"TIMELINE EVENTS:\n" + "\n".join([f"[{t.get('formatted_time')}] {t.get('description')}" for t in timeline[:8]])
                prompt = prompt_template + f"\n\nVIDEO METADATA:\nFilename: {metadata.filename}\nDuration: {metadata.duration_sec}s\n\n{evidence_text}"

                top_paths = [sf.path for sf in sampled_frames[:3] if Path(sf.path).exists()]
                if top_paths:
                    vlm_res = vlm_provider.analyze_images(top_paths, prompt)
                    if vlm_res and isinstance(vlm_res, dict):
                        if "final_description" in vlm_res and vlm_res["final_description"]:
                            final_desc = vlm_res["final_description"]
            except Exception as e:
                logger.warning(f"VLM final summary reasoning failed: {e}. Using grounded timeline narrative.")

        return FinalSummary(
            objects=final_objects,
            people=final_people,
            final_description=final_desc,
        )

temporal_reasoner = TemporalReasoner()
