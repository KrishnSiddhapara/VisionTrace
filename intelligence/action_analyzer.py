from typing import List, Dict, Any
from models.schemas import FrameObservation, TrackedObject

class ActionAnalyzer:
    """Infers higher-level activities from low-level movement and VLM observations."""

    def analyze_actions(self, frame_observations: List[FrameObservation], tracks: List[TrackedObject]) -> Dict[str, Any]:
        activity_summary = {}
        for obs in frame_observations:
            for act in obs.activities:
                activity_summary[act] = activity_summary.get(act, 0) + 1

        interaction_summary = {}
        for obs in frame_observations:
            for inter in obs.interactions:
                interaction_summary[inter] = interaction_summary.get(inter, 0) + 1

        return {
            "frequent_activities": activity_summary,
            "frequent_interactions": interaction_summary,
            "total_tracked_entities": len(tracks),
        }

action_analyzer = ActionAnalyzer()
