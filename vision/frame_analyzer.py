from pathlib import Path
from typing import Union, Dict, Any, List

from config.settings import settings
from models.schemas import SampledFrame, FrameObservation, PersonObservation, ObjectObservation
from vision.vlm import get_vlm_provider, VLMProvider
from utils.logger import logger
from utils.caching import cache_manager

class FrameAnalyzer:
    """Analyzes sampled frames using structured VLM output and multi-frame window context."""

    def __init__(self, vlm_provider: VLMProvider = None):
        self.vlm = vlm_provider or get_vlm_provider()
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        prompt_file = settings.PROMPTS_DIR / "frame_analysis.txt"
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")
        return "Analyze this video frame and return a structured JSON of environment, people, objects, and activities."

    def analyze_frame(self, sampled_frame: SampledFrame, prev_obs: FrameObservation = None) -> FrameObservation:
        path = Path(sampled_frame.path)
        cache_key = f"frame_vlm_{sampled_frame.frame_id}_{path.name}"

        cached = cache_manager.get(cache_key)
        if cached:
            return FrameObservation(**cached)

        logger.info(f"Analyzing frame {sampled_frame.frame_id} (timestamp: {sampled_frame.timestamp}s) with VLM...")

        # Construct prompt with optional temporal context from previous frame
        prompt = self.prompt_template
        if prev_obs and prev_obs.observations:
            context_str = "\n".join(prev_obs.observations[:3])
            prompt += f"\n\n[TEMPORAL CONTEXT FROM PREVIOUS FRAME at {prev_obs.timestamp}s]:\n{context_str}\nDescribe any visual state changes compared to the previous frame."

        raw_json = self.vlm.analyze_image(path, prompt)

        # Parse & sanitize raw output into FrameObservation Pydantic model
        people_obs = []
        for p in raw_json.get("people", []):
            if isinstance(p, dict):
                people_obs.append(
                    PersonObservation(
                        temporary_id=p.get("temporary_id"),
                        description=p.get("description", "Person detected"),
                        activity=p.get("activity"),
                        location=p.get("location"),
                        confidence=float(p.get("confidence", 0.9)),
                    )
                )

        objects_obs = []
        for o in raw_json.get("objects", []):
            if isinstance(o, dict):
                objects_obs.append(
                    ObjectObservation(
                        name=o.get("name", "object"),
                        description=o.get("description"),
                        location=o.get("location"),
                        confidence=float(o.get("confidence", 0.9)),
                    )
                )

        observation = FrameObservation(
            frame_id=sampled_frame.frame_id,
            timestamp=sampled_frame.timestamp,
            scene_id=sampled_frame.scene_id,
            environment=raw_json.get("environment", "Unknown environment"),
            people=people_obs,
            objects=objects_obs,
            activities=raw_json.get("activities", []),
            interactions=raw_json.get("interactions", []),
            relationships=raw_json.get("relationships", []),
            visible_text=raw_json.get("visible_text", []),
            observations=raw_json.get("observations", []),
            uncertainties=raw_json.get("uncertainties", []),
        )

        # Save to cache
        cache_manager.set(cache_key, observation.model_dump())
        return observation

frame_analyzer = FrameAnalyzer()
