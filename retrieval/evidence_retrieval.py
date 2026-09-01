from typing import List, Dict, Any
from models.schemas import VideoMemory

class EvidenceRetriever:
    """Retrieves specific frame evidence, tracks, state changes, and timestamps for Q&A answers."""

    def retrieve_evidence(self, memory: VideoMemory, timestamps: List[float]) -> Dict[str, Any]:
        matched_frames = []
        matched_paths = []
        matched_facts = []

        # Map sampled frames by frame_id for path lookup
        frame_paths_map = {f.frame_id: f.path for f in memory.sampled_frames}

        for ts in timestamps:
            for obs in memory.frame_observations:
                if abs(obs.timestamp - ts) <= 3.5:
                    matched_frames.append(obs.frame_id)
                    if obs.frame_id in frame_paths_map:
                        matched_paths.append(frame_paths_map[obs.frame_id])
                    
                    facts_str = ", ".join(obs.activities + obs.interactions + obs.confirmed_changes or ["Frame observed"])
                    matched_facts.append(f"At {obs.timestamp:.1f}s: {facts_str}")

        # Retrieve relevant tracked entities
        relevant_tracks = [
            f"{trk.track_id} ({trk.object_type}) first seen {trk.first_seen:.1f}s, last seen {trk.last_seen:.1f}s"
            for trk in memory.tracks
        ]

        return {
            "evidence_frames": list(set(matched_frames)),
            "evidence_paths": list(set(matched_paths)),
            "observed_facts": list(set(matched_facts)),
            "relevant_tracks": relevant_tracks,
        }

evidence_retriever = EvidenceRetriever()
