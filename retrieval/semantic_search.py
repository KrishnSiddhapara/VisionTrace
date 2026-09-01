from typing import List, Dict, Any
from models.schemas import VideoMemory
from retrieval.embeddings import embeddings_engine
from utils.logger import logger

class SemanticSearchEngine:
    """Performs semantic queries against structured video memory."""

    def search(self, memory: VideoMemory, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vec = embeddings_engine.get_embedding(query)
        results = []

        # Search across frame observations
        for obs in memory.frame_observations:
            text = f"{obs.environment} {' '.join(obs.activities)} {' '.join(obs.relationships)} {' '.join(obs.observations)}"
            obs_vec = embeddings_engine.get_embedding(text)
            score = embeddings_engine.cosine_similarity(query_vec, obs_vec)

            if score > 0.05:
                results.append({
                    "timestamp": obs.timestamp,
                    "frame_id": obs.frame_id,
                    "score": round(score, 3),
                    "text": text,
                    "type": "Frame Observation",
                })

        # Search across events
        for evt in memory.events:
            text = f"{evt.event_type} {evt.subject or ''} {evt.object or ''} {evt.description}"
            evt_vec = embeddings_engine.get_embedding(text)
            score = embeddings_engine.cosine_similarity(query_vec, evt_vec)

            if score > 0.05:
                results.append({
                    "timestamp": evt.start_time,
                    "score": round(score, 3),
                    "text": text,
                    "type": f"Event ({evt.event_type})",
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

semantic_search_engine = SemanticSearchEngine()
