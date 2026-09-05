from pathlib import Path
from typing import Union, Dict, Any, List, Optional

from config.settings import settings
from models.schemas import SampledFrame, FrameObservation, PersonObservation, ObjectObservation
from vision.vlm import get_vlm_provider, VLMProvider
from vision.quality import frame_quality_checker
from utils.logger import logger
from utils.caching import cache_manager

class FrameAnalyzer:
    """Analyzes sampled frames using structured VLM output, temporal windows, and visual quality evaluation."""

    def __init__(self, vlm_provider: VLMProvider = None):
        self.vlm = vlm_provider or get_vlm_provider()
        self.single_prompt_template = self._load_prompt("frame_analysis.txt")
        self.window_prompt_template = self._load_prompt("temporal_window.txt")

    def _load_prompt(self, filename: str) -> str:
        prompt_file = settings.PROMPTS_DIR / filename
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")
        return ""

    def analyze_frame_window(
        self,
        sampled_frame: SampledFrame,
        prev_frame: Optional[SampledFrame] = None,
        next_frame: Optional[SampledFrame] = None,
        video_hash: str = ""
    ) -> FrameObservation:
        """
        Analyze a temporal window centered on `sampled_frame` [PREV, CURR, NEXT].
        Incorporates visual quality checks, versioned caching, and retry handling.
        """
        path = Path(sampled_frame.path)
        if not path.exists():
            return FrameObservation(
                frame_id=sampled_frame.frame_id,
                timestamp=sampled_frame.timestamp,
                scene_id=sampled_frame.scene_id,
                environment="Unavailable",
                uncertainties=["Frame image file missing"],
                evidence_strength="UNAVAILABLE",
                is_analyzed=False,
            )

        # 1. Quality evaluation
        quality_info = frame_quality_checker.evaluate_quality(path)
        content_hash = quality_info.get("content_hash", "")
        sampled_frame.quality_score = quality_info.get("quality_score", 1.0)
        sampled_frame.is_blurry = quality_info.get("is_blurry", False)
        sampled_frame.content_hash = content_hash

        # 2. Versioned Cache Key
        cache_key = cache_manager.build_versioned_key(
            prefix=f"frame_vlm_{sampled_frame.frame_id}",
            video_hash=video_hash,
            content_hash=content_hash,
        )

        cached = cache_manager.get(cache_key)
        if cached:
            return FrameObservation(**cached)

        logger.info(f"Analyzing frame window {sampled_frame.frame_id} (timestamp: {sampled_frame.timestamp}s, quality: {sampled_frame.quality_score})...")

        # 3. Build temporal window frame list & prompt
        window_frames = []
        if prev_frame and Path(prev_frame.path).exists():
            window_frames.append(prev_frame)
        window_frames.append(sampled_frame)
        if next_frame and Path(next_frame.path).exists():
            window_frames.append(next_frame)

        if len(window_frames) > 1 and self.window_prompt_template:
            p_ts = prev_frame.timestamp if prev_frame else sampled_frame.timestamp
            c_ts = sampled_frame.timestamp
            n_ts = next_frame.timestamp if next_frame else sampled_frame.timestamp
            prompt = (
                self.window_prompt_template
                .replace("{prev_timestamp}", f"{p_ts:.2f}")
                .replace("{curr_timestamp}", f"{c_ts:.2f}")
                .replace("{next_timestamp}", f"{n_ts:.2f}")
            )
            image_paths = [f.path for f in window_frames]
            raw_json = self.vlm.analyze_images(image_paths, prompt)
        else:
            prompt = self.single_prompt_template
            raw_json = self.vlm.analyze_image(path, prompt)

        # 4. Check if VLM succeeded or returned empty result
        if not raw_json:
            obs = FrameObservation(
                frame_id=sampled_frame.frame_id,
                timestamp=sampled_frame.timestamp,
                scene_id=sampled_frame.scene_id,
                environment="Unanalyzed",
                uncertainties=["VLM request returned empty response or failed retries"],
                evidence_strength="UNAVAILABLE",
                quality_score=sampled_frame.quality_score,
                is_analyzed=False,
            )
            return obs

        # 5. Parse people observations
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

        # 6. Parse object observations
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

        env_str = raw_json.get("environment") or "Unknown environment"
        evidence_str = raw_json.get("evidence_strength") or "HIGH"
        if sampled_frame.is_blurry or sampled_frame.quality_score < 0.6:
            evidence_str = "MEDIUM" if evidence_str == "HIGH" else "LOW"

        def _ensure_list(val: Any) -> List[str]:
            if isinstance(val, list):
                return [str(item) for item in val if item is not None]
            elif isinstance(val, str) and val.strip():
                return [val.strip()]
            return []

        observation = FrameObservation(
            frame_id=sampled_frame.frame_id,
            timestamp=sampled_frame.timestamp,
            scene_id=sampled_frame.scene_id,
            environment=str(env_str),
            people=people_obs,
            objects=objects_obs,
            activities=_ensure_list(raw_json.get("activities")),
            interactions=_ensure_list(raw_json.get("interactions")),
            relationships=_ensure_list(raw_json.get("relationships")),
            visible_text=_ensure_list(raw_json.get("visible_text")),
            observations=_ensure_list(raw_json.get("observations")),
            uncertainties=_ensure_list(raw_json.get("uncertainties")),
            confirmed_changes=_ensure_list(raw_json.get("confirmed_changes")),
            possible_changes=_ensure_list(raw_json.get("possible_changes")),
            evidence_strength=str(evidence_str),
            quality_score=sampled_frame.quality_score,
            is_analyzed=True,
            motion_score=sampled_frame.motion_score,
            selection_reason=sampled_frame.selection_reason,
        )

        # Cache valid result
        cache_manager.set(cache_key, observation.model_dump())
        return observation

    def analyze_frame(self, sampled_frame: SampledFrame, prev_obs: FrameObservation = None) -> FrameObservation:
        """Backward compatible single frame interface."""
        return self.analyze_frame_window(sampled_frame)

frame_analyzer = FrameAnalyzer()
