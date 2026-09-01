from typing import List, Dict, Any
from models.schemas import Scene, VideoEvent, TrackedObject, FrameObservation, VideoMetadata
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

        # Filter and sort verified events by start_time
        sorted_events = sorted(events, key=lambda e: (e.start_time, e.event_id))

        for idx, item in enumerate(sorted_events, start=1):
            m, s = divmod(item.start_time, 60)
            time_str = f"{int(m):02d}:{s:05.2f}"

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
                "evidence_frames": item.evidence_frames,
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

        # Quick Summary (1-2 sentences)
        quick = (
            f"Video '{metadata.filename}' ({dur:.1f}s, {sc_cnt} scenes) features {trk_cnt} tracked entities "
            f"and {len(timeline)} verified visual events."
        )

        # Standard Summary (Structured paragraph)
        major_events = [t['description'] for t in timeline if t.get('confidence', 0) >= 0.85][:5]
        event_highlights = "; ".join(major_events) if major_events else "continuous scene activity"

        standard = (
            f"The video '{metadata.filename}' spans {dur:.1f} seconds across {sc_cnt} distinct visual scenes. "
            f"A total of {trk_cnt} distinct entities were tracked with spatial BBox IoU continuity. "
            f"Primary visual progression includes: {event_highlights}."
        )

        # Detailed Summary (Chronological breakdown)
        detailed_lines = [
            f"- [{t['formatted_time']}] {t['description']} (Confidence: {int(t['confidence']*100)}%)"
            for t in timeline
        ]
        detailed = "\n".join(detailed_lines) if detailed_lines else "No specific events logged."

        # Technical Analysis
        technical = (
            f"Technical & Quality Metrics:\n"
            f"- Resolution: {metadata.resolution_str} @ {metadata.fps} FPS\n"
            f"- Duration: {metadata.duration_sec}s ({metadata.frame_count} total frames)\n"
            f"- Codec: {metadata.codec}\n"
            f"- PySceneDetect Scenes: {sc_cnt}\n"
            f"- Tracked Entity Trajectories: {trk_cnt}\n"
            f"- Verified Timeline Events: {len(timeline)}\n"
        )

        return {
            "quick": quick,
            "standard": standard,
            "detailed": detailed,
            "technical": technical,
        }

temporal_reasoner = TemporalReasoner()
