from pathlib import Path
from typing import Dict, Any, List
from models.schemas import VideoMemory, QAResponse
from retrieval.semantic_search import semantic_search_engine
from retrieval.evidence_retrieval import evidence_retriever
from vision.vlm import get_vlm_provider, VLMProvider
from config.settings import settings
from utils.logger import logger

class VideoQAEngine:
    """Answers natural language questions about video using evidence retrieval and VLM reasoning."""

    def __init__(self, vlm_provider: VLMProvider = None):
        self.vlm = vlm_provider or get_vlm_provider()
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        prompt_file = settings.PROMPTS_DIR / "video_qa.txt"
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")
        return ""

    def answer_question(self, memory: VideoMemory, question: str) -> QAResponse:
        logger.info(f"Answering video Q&A query: '{question}'")
        search_results = semantic_search_engine.search(memory, question, top_k=4)

        if not search_results:
            return QAResponse(
                question=question,
                answer="Insufficient visual evidence to determine this.",
                confidence=0.30,
                evidence_timestamps=[],
                evidence_frames=[],
                observed_facts=[],
                inferred_facts=[],
                unknown_aspects=["No matching visual evidence found in video memory."],
            )

        evidence_ts = [r["timestamp"] for r in search_results]
        evidence = evidence_retriever.retrieve_evidence(memory, evidence_ts)
        evidence_paths = [p for p in evidence.get("evidence_paths", []) if Path(p).exists()]

        # Prepare evidence prompt text
        facts_summary = "\n".join(evidence.get("observed_facts", [])[:6])
        tracks_summary = "\n".join(evidence.get("relevant_tracks", [])[:5])
        evidence_text = f"OBSERVED FACTS:\n{facts_summary}\n\nTRACKED ENTITIES:\n{tracks_summary}"

        if self.prompt_template and (evidence_paths or not settings.VLM_MOCK_MODE):
            prompt = (
                self.prompt_template
                .replace("{question}", question)
                .replace("{retrieved_evidence}", evidence_text)
            )
            vlm_res = self.vlm.analyze_images(evidence_paths[:3], prompt) if evidence_paths else self.vlm.analyze_image(memory.sampled_frames[0].path if memory.sampled_frames else "", prompt)
            
            if vlm_res and "answer" in vlm_res:
                ans_text = vlm_res.get("answer", "")
                conf = float(vlm_res.get("confidence", search_results[0]["score"]))
                obs = vlm_res.get("observed_facts", evidence.get("observed_facts", []))
                inf = vlm_res.get("inferred_facts", ["Visual progression supports state continuity."])
                unk = vlm_res.get("unknown_aspects", ["Subjective human intent is unobservable."])

                return QAResponse(
                    question=question,
                    answer=ans_text,
                    confidence=round(conf, 2),
                    evidence_timestamps=evidence_ts,
                    evidence_frames=evidence.get("evidence_frames", []),
                    observed_facts=obs,
                    inferred_facts=inf,
                    unknown_aspects=unk,
                )

        # Fallback grounded response
        top = search_results[0]
        answer_str = f"Based on visual evidence near timestamp {top['timestamp']:.1f}s: {top['text']}"

        return QAResponse(
            question=question,
            answer=answer_str,
            confidence=round(top["score"], 2),
            evidence_timestamps=evidence_ts,
            evidence_frames=evidence.get("evidence_frames", []),
            observed_facts=evidence.get("observed_facts", []),
            inferred_facts=["Visual evidence supports observed timestamp activity."],
            unknown_aspects=["Subjective human intent cannot be determined from visual evidence alone."],
        )

video_qa_engine = VideoQAEngine()
