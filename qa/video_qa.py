from typing import Dict, Any
from models.schemas import VideoMemory, QAResponse
from retrieval.semantic_search import semantic_search_engine
from retrieval.evidence_retrieval import evidence_retriever
from utils.logger import logger

class VideoQAEngine:
    """Answers natural language questions about video using structured video memory."""

    def answer_question(self, memory: VideoMemory, question: str) -> QAResponse:
        search_results = semantic_search_engine.search(memory, question, top_k=3)

        evidence_ts = [r["timestamp"] for r in search_results]
        evidence = evidence_retriever.retrieve_evidence(memory, evidence_ts)

        if search_results:
            top = search_results[0]
            answer = f"Based on visual evidence around timestamp {top['timestamp']}s: {top['text']}"
            confidence = top["score"]
        else:
            answer = "No direct visual evidence was found in the video memory for this question."
            confidence = 0.40

        observed = evidence.get("observed_facts", [])
        inferred = ["Visual progression indicates continuous scene context."] if search_results else []
        unknown = ["Exact subjective intent cannot be determined from visual evidence alone."]

        return QAResponse(
            question=question,
            answer=answer,
            confidence=round(confidence, 2),
            evidence_timestamps=evidence_ts,
            evidence_frames=evidence.get("evidence_frames", []),
            observed_facts=observed,
            inferred_facts=inferred,
            unknown_aspects=unknown,
        )

video_qa_engine = VideoQAEngine()
