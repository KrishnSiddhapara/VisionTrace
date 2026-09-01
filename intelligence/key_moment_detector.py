from typing import List, Dict, Any
from models.schemas import VideoEvent, TrackedObject, Scene
from utils.logger import logger

class KeyMomentDetector:
    """Ranks important moments in the video using importance scoring."""

    def detect_key_moments(self, events: List[VideoEvent], tracks: List[TrackedObject], scenes: List[Scene]) -> List[Dict[str, Any]]:
        key_moments = []

        # Start of video
        if scenes:
            key_moments.append({
                "timestamp": scenes[0].start_time,
                "title": "Video Start & Scene Init",
                "importance_score": 0.95,
                "description": f"Initial scene starts at {scenes[0].start_time}s",
            })

        # Track appearances
        for trk in tracks:
            key_moments.append({
                "timestamp": trk.first_seen,
                "title": f"Entity Appearance: {trk.track_id}",
                "importance_score": 0.88,
                "description": f"{trk.track_id} first detected in video at {trk.first_seen}s",
            })

        # Sort by timestamp
        key_moments.sort(key=lambda x: x["timestamp"])

        logger.info(f"Identified {len(key_moments)} key video moments.")
        return key_moments

key_moment_detector = KeyMomentDetector()
