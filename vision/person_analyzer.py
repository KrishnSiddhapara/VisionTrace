from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

from config.settings import settings
from models.schemas import (
    PersonEntity, PersonSpecificObservation, PersonActionRecord,
    SampledFrame, YOLODetection, FrameObservation
)
from vision.vlm import get_vlm_provider, VLMProvider
from vision.person_visual_grounder import person_visual_grounder
from utils.logger import logger
from utils.caching import cache_manager


class PersonAnalyzer:
    """
    Executes isolated, grounded VLM analysis for individual canonical PersonEntities.
    Guarantees that actions, held objects, and activities belonging to one person
    are NEVER assigned, leaked, or copied to another person entity.
    """

    def __init__(self, vlm_provider: VLMProvider = None):
        self.vlm = vlm_provider or get_vlm_provider()
        self.prompt_template = self._load_prompt("person_analysis.txt")

    def _load_prompt(self, filename: str) -> str:
        prompt_file = settings.PROMPTS_DIR / filename
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")
        return ""

    def analyze_canonical_person(
        self,
        person_entity: PersonEntity,
        sampled_frames: List[SampledFrame],
        video_hash: str = ""
    ) -> List[PersonSpecificObservation]:
        """
        Analyze representative keyframes for a single target person using isolated Evidence A (crop)
        and Evidence B (highlighted frame) inputs.
        """
        if not person_entity.positions or not sampled_frames:
            return []

        # Find frames where target person was detected
        sf_by_ts: Dict[float, SampledFrame] = {round(sf.timestamp, 2): sf for sf in sampled_frames}
        
        person_pos_by_ts: Dict[float, List[float]] = {}
        for pos in person_entity.positions:
            ts = round(float(pos.get("timestamp", 0.0)), 2)
            person_pos_by_ts[ts] = pos.get("bbox", [0, 0, 0, 0])

        matching_timestamps = sorted(list(set(sf_by_ts.keys()).intersection(set(person_pos_by_ts.keys()))))
        if not matching_timestamps:
            # Fallback: pick closest sampled frame timestamp
            for pos in person_entity.positions:
                p_ts = float(pos.get("timestamp", 0.0))
                closest_sf = min(sampled_frames, key=lambda sf: abs(sf.timestamp - p_ts))
                if abs(closest_sf.timestamp - p_ts) <= 2.0:
                    matching_timestamps.append(round(closest_sf.timestamp, 2))
                    sf_by_ts[round(closest_sf.timestamp, 2)] = closest_sf
                    person_pos_by_ts[round(closest_sf.timestamp, 2)] = pos.get("bbox", [0, 0, 0, 0])
                    break

        if not matching_timestamps:
            return []

        # Limit to max representative frames per person (e.g. 2 keyframes: start/middle)
        max_frames_per_person = getattr(settings, "PERSON_VLM_MAX_FRAMES_PER_PERSON", 2)
        if len(matching_timestamps) > max_frames_per_person:
            selected_ts = [matching_timestamps[0], matching_timestamps[-1]]
            if max_frames_per_person >= 3 and len(matching_timestamps) >= 3:
                selected_ts.insert(1, matching_timestamps[len(matching_timestamps) // 2])
        else:
            selected_ts = matching_timestamps[:max_frames_per_person]

        out_dir = settings.PROCESSED_DIR / video_hash / "person_grounding"
        observations: List[PersonSpecificObservation] = []

        for ts in selected_ts:
            sf = sf_by_ts[ts]
            bbox = person_pos_by_ts[ts]
            image_path = sf.path

            if not Path(image_path).exists():
                continue

            # 1. Generate Evidence A (crop) and Evidence B (highlighted frame)
            crop_path, highlight_path = person_visual_grounder.generate_person_evidence_images(
                image_path=image_path,
                bbox=bbox,
                person_id=person_entity.person_id,
                out_dir=out_dir
            )

            if not crop_path or not highlight_path:
                continue

            # 2. Versioned Cache Key per Person per Frame
            bbox_str = "_".join([f"{v:.0f}" for v in bbox])
            cache_key = cache_manager.build_versioned_key(
                prefix=f"person_vlm_{person_entity.person_id}_{sf.frame_id}_{bbox_str}",
                video_hash=video_hash,
            )

            cached = cache_manager.get(cache_key)
            if cached:
                logger.info(f"[Person VLM Cache HIT] Reusing isolated observation for {person_entity.person_id} at {ts:.2f}s.")
                observations.append(PersonSpecificObservation(**cached))
                continue

            # 3. Prompt Formatting
            prompt = (
                self.prompt_template
                .replace("{person_id}", person_entity.person_id)
                .replace("{timestamp}", f"{ts:.2f}")
            )

            # 4. VLM Provider Call with Evidence A (crop) and Evidence B (highlighted frame)
            raw_json = self.vlm.analyze_images([crop_path, highlight_path], prompt)
            if not raw_json or not isinstance(raw_json, dict):
                continue

            # 5. Parse Grounded JSON Response
            actions_list: List[PersonActionRecord] = []
            for act_dict in raw_json.get("actions", []):
                if isinstance(act_dict, dict):
                    actions_list.append(
                        PersonActionRecord(
                            action=str(act_dict.get("action", "observing")).strip(),
                            object_name=act_dict.get("object"),
                            confidence=float(act_dict.get("confidence", 0.90)),
                            evidence=str(act_dict.get("evidence", "")).strip(),
                            timestamp=ts,
                        )
                    )

            objects_held: List[Dict[str, Any]] = []
            for obj_item in raw_json.get("objects_held", []):
                if isinstance(obj_item, dict) and obj_item.get("object"):
                    o_name = str(obj_item["object"]).strip().lower()
                    if o_name not in ("none", "null", "nothing", "no clearly visible held object"):
                        objects_held.append({
                            "object": o_name,
                            "confidence": float(obj_item.get("confidence", 0.90))
                        })

            obs = PersonSpecificObservation(
                canonical_person_id=person_entity.person_id,
                frame_id=sf.frame_id,
                timestamp=ts,
                bbox=bbox,
                clothing=raw_json.get("clothing"),
                actions=actions_list,
                objects_held=objects_held,
                interactions=[str(i) for i in raw_json.get("interactions", []) if i],
                uncertainties=[str(u) for u in raw_json.get("uncertainties", []) if u],
                confidence=0.92,
                evidence_source="PERSON_VLM_GROUNDED"
            )

            obs_dict = obs.model_dump()
            cache_manager.set(cache_key, obs_dict)
            observations.append(obs)

            logger.info(
                f"[GROUNDED PERSON VLM] {person_entity.person_id} @ {ts:.2f}s | "
                f"Actions: {[a.action for a in actions_list]} | Held Objects: {[oh['object'] for oh in objects_held]}"
            )

        return observations


person_analyzer = PersonAnalyzer()
