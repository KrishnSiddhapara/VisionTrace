from typing import List, Dict, Any
from models.schemas import VideoMemory

class EvidenceRetriever:
    """Retrieves specific frame evidence and facts for Q&A answers."""

    def retrieve_evidence(self, memory: VideoMemory, timestamps: List[float]) -> Dict[str, Any]:
        matched_frames = []
        matched_facts = []

        for ts in timestamps:
            for obs in memory.frame_observations:
                if abs(obs.timestamp - ts) <= 3.0:
                    matched_frames.append(obs.frame_id)
                    matched_facts.append(f"At {obs.timestamp}s: {', '.join(obs.activities or ['Frame observed'])}")

        return {
            "evidence_frames": list(set(matched_frames)),
            "observed_facts": list(set(matched_facts)),
        }

evidence_retriever = EvidenceRetriever()
