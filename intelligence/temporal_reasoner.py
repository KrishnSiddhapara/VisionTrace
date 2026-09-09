from typing import List, Dict, Any
from pathlib import Path
from models.schemas import (
    Scene, VideoEvent, TrackedObject, FrameObservation, VideoMetadata,
    SampledFrame, FinalSummary, FinalObjectRecord, FinalPersonRecord
)
from vision.vlm import get_vlm_provider
from vision.tracker import normalize_class_name
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
        # 1. Deduplicate & Aggregate Objects using canonical PhysicalObjectRegistry
        from vision.object_registry import physical_object_registry
        physical_objects = physical_object_registry.reconcile_tracks(tracks, frame_observations)

        final_objects = []
        covered_categories = set()

        non_person_phys = [
            po for po in physical_objects
            if po.canonical_name != "person"
        ]

        for po in non_person_phys:
            m_s, s_s = divmod(po.first_seen, 60)
            m_e, s_e = divmod(po.last_seen, 60)
            covered_categories.add(po.canonical_name)

            display_name = "Ball" if po.canonical_name == "sports ball" else po.object_id
            final_objects.append(
                FinalObjectRecord(
                    name=display_name,
                    description=f"{display_name} physical object trajectory ({', '.join(po.track_ids)}).",
                    first_seen=f"{int(m_s):02d}:{s_s:04.1f}",
                    last_seen=f"{int(m_e):02d}:{s_e:04.1f}",
                    movement=", ".join(po.lifecycle_events) if po.lifecycle_events else "Observed in scene",
                    state_changes=[],
                    interactions=po.interactions,
                    confidence=round(po.avg_confidence, 2),
                )
            )

        # Also pull objects from VLM frame observations ONLY if category not covered by tracks & is a main object
        from vision.tracker import is_main_object
        vlm_added_cats = set()
        for obs in frame_observations:
            for o in obs.objects:
                norm_cat = normalize_class_name(o.name)
                has_change = any(o.name.lower() in item.lower() for item in obs.confirmed_changes + obs.interactions)
                if norm_cat != "person" and norm_cat not in covered_categories and norm_cat not in vlm_added_cats:
                    if is_main_object(o.name, has_movement_or_interaction=has_change):
                        m_s, s_s = divmod(obs.timestamp, 60)
                        display_name = o.name.capitalize()
                        vlm_added_cats.add(norm_cat)
                        final_objects.append(
                            FinalObjectRecord(
                                name=display_name,
                                description=o.description or f"{display_name} visible in frame keyframes",
                                first_seen=f"{int(m_s):02d}:{s_s:04.1f}",
                                last_seen=f"{int(m_s):02d}:{s_s:04.1f}",
                                movement="Observed in frame keyframes",
                                state_changes=obs.confirmed_changes,
                                interactions=obs.interactions,
                                confidence=round(float(o.confidence or 0.88), 2),
                            )
                        )

        # 2. Aggregate People Entities using CanonicalPersonRegistry (Single Source of Truth)
        from vision.person_registry import canonical_person_registry
        person_entities = canonical_person_registry.reconcile_person_tracks(tracks, frame_observations, metadata)
        final_people = []

        for pe in person_entities:
            m_s, s_s = divmod(pe.first_seen, 60)
            m_e, s_e = divmod(pe.last_seen, 60)

            short_note_str = pe.get_shortnote_description()

            final_people.append(
                FinalPersonRecord(
                    temporary_id=pe.person_id,
                    description=short_note_str,
                    first_seen=f"{int(m_s):02d}:{s_s:04.1f}",
                    last_seen=f"{int(m_e):02d}:{s_e:04.1f}",
                    activities=pe.activities,
                    movements=pe.lifecycle_events or [f"{pe.motion_state.capitalize()} stance"],
                    interactions=pe.interactions,
                    confidence=round(pe.avg_confidence, 2),
                )
            )


        # 3. Generate Chronological Final Description
        chronological_events = [t for t in timeline if t.get("description")]
        if chronological_events:
            lines = [f"📌 **Overview**: Video '{metadata.filename}' spans {metadata.duration_sec:.1f}s across {len(scenes)} scenes."]
            for item in chronological_events:
                t_str = item.get("formatted_time", f"{item.get('timestamp', 0):.1f}s")
                desc = item.get("description", "")
                lines.append(f"⏱️ **[{t_str}]**: {desc}")
            final_desc = "\n\n".join(lines)
        else:
            final_desc = f"📌 **Overview**: Video '{metadata.filename}' spans {metadata.duration_sec:.1f}s.\n\n⚠️ Insufficient visual change events were detected to construct a detailed movement narrative."

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
