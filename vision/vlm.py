import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Union, List
from PIL import Image

from config.settings import settings
from utils.logger import logger

def parse_vlm_json_response(raw_text: str) -> Dict[str, Any]:
    """
    Robust JSON parser for VLM text output.
    Strips markdown code blocks, extracts JSON substrings, and validates dict structure.
    """
    if not raw_text:
        return {}

    cleaned = raw_text.strip()
    # Strip markdown fences ```json ... ```
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1:
        json_str = cleaned[start : end + 1]
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode warning on VLM output: {e}")

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {}


class VLMProvider(ABC):
    """Abstract Vision-Language Model interface."""

    @abstractmethod
    def analyze_image(self, image_path: Union[str, Path], prompt: str) -> Dict[str, Any]:
        """Analyze an image with prompt and return parsed JSON dict response."""
        pass

    def analyze_images(self, image_paths: List[Union[str, Path]], prompt: str) -> Dict[str, Any]:
        """Analyze multiple consecutive frames with prompt for temporal comparison."""
        # Default fallback: analyze middle frame if multi-image not specialized
        if not image_paths:
            return {}
        mid_idx = len(image_paths) // 2
        return self.analyze_image(image_paths[mid_idx], prompt)


class GeminiVLMProvider(VLMProvider):
    """Google Gemini VLM Provider implementation with retry logic and strict grounding."""

    def __init__(self, api_key: str = "", model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or settings.VLM_API_KEY
        self.model_name = model_name or settings.VLM_MODEL
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized Gemini VLM provider with model {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")

    def analyze_image(self, image_path: Union[str, Path], prompt: str) -> Dict[str, Any]:
        return self.analyze_images([image_path], prompt)

    def analyze_images(self, image_paths: List[Union[str, Path]], prompt: str) -> Dict[str, Any]:
        if not self.client:
            if settings.VLM_MOCK_MODE or settings.VLM_PROVIDER == "mock":
                return MockVLMProvider().analyze_image(image_paths[0], prompt)
            logger.warning(f"[VLM Telemetry] Gemini client not initialized (API Key missing or invalid) and VLM_MOCK_MODE=False. Frame marked unanalyzed.")
            return {}

        paths = [Path(p) for p in image_paths if Path(p).exists()]
        if not paths:
            return {}

        max_retries = settings.VLM_MAX_RETRIES
        for attempt in range(1, max_retries + 1):
            start_t = time.time()
            try:
                images = [Image.open(p).convert("RGB") for p in paths]
                contents = images + [prompt]
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                )
                duration = round(time.time() - start_t, 2)
                raw_text = response.text
                parsed = parse_vlm_json_response(raw_text)
                if parsed:
                    logger.info(
                        f"[VLM Telemetry] SUCCESS | Model: {self.model_name} | Frames: {[p.name for p in paths]} | "
                        f"Attempt: {attempt}/{max_retries} | Duration: {duration}s | Status: PARSED_OK"
                    )
                    return parsed

                logger.warning(
                    f"[VLM Telemetry] JSON_PARSE_FAIL | Model: {self.model_name} | Frames: {[p.name for p in paths]} | "
                    f"Attempt: {attempt}/{max_retries} | Duration: {duration}s"
                )
            except Exception as e:
                duration = round(time.time() - start_t, 2)
                logger.warning(
                    f"[VLM Telemetry] REQUEST_FAIL | Model: {self.model_name} | Frames: {[p.name for p in paths]} | "
                    f"Attempt: {attempt}/{max_retries} | Duration: {duration}s | Error: {e}"
                )

            if attempt < max_retries:
                time.sleep(0.5 * attempt)

        if settings.VLM_MOCK_MODE or settings.VLM_PROVIDER == "mock":
            logger.warning("[VLM Telemetry] Retries exhausted. Explicit Mock Mode enabled -> returning MockVLM.")
            return MockVLMProvider().analyze_image(paths[0], prompt)

        logger.error(f"[VLM Telemetry] FAILED_ALL_RETRIES | Frames: {[p.name for p in paths]} | Frame marked unavailable.")
        return {}


class OpenAIVLMProvider(VLMProvider):
    """OpenAI Vision VLM Provider implementation with retry logic and strict grounding."""

    def __init__(self, api_key: str = "", model_name: str = "gpt-4o-mini", base_url: str = ""):
        self.api_key = api_key or settings.VLM_API_KEY
        self.model_name = model_name or settings.VLM_MODEL
        self.base_url = base_url or settings.VLM_BASE_URL
        self.client = None

        if self.api_key:
            try:
                from openai import OpenAI
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self.client = OpenAI(**kwargs)
                logger.info(f"Initialized OpenAI VLM provider with model {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")

    def analyze_image(self, image_path: Union[str, Path], prompt: str) -> Dict[str, Any]:
        return self.analyze_images([image_path], prompt)

    def analyze_images(self, image_paths: List[Union[str, Path]], prompt: str) -> Dict[str, Any]:
        if not self.client:
            if settings.VLM_MOCK_MODE or settings.VLM_PROVIDER == "mock":
                return MockVLMProvider().analyze_image(image_paths[0], prompt)
            logger.warning("[VLM Telemetry] OpenAI client not initialized and VLM_MOCK_MODE=False. Frame marked unanalyzed.")
            return {}

        import base64
        paths = [Path(p) for p in image_paths if Path(p).exists()]
        if not paths:
            return {}

        max_retries = settings.VLM_MAX_RETRIES
        for attempt in range(1, max_retries + 1):
            start_t = time.time()
            try:
                content_payload = [{"type": "text", "text": prompt}]
                for p in paths:
                    with open(p, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    content_payload.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content_payload}],
                    response_format={"type": "json_object"},
                )
                duration = round(time.time() - start_t, 2)
                content = response.choices[0].message.content
                parsed = parse_vlm_json_response(content)
                if parsed:
                    logger.info(
                        f"[VLM Telemetry] SUCCESS | Model: {self.model_name} | Frames: {[p.name for p in paths]} | "
                        f"Attempt: {attempt}/{max_retries} | Duration: {duration}s | Status: PARSED_OK"
                    )
                    return parsed

                logger.warning(
                    f"[VLM Telemetry] JSON_PARSE_FAIL | Model: {self.model_name} | Frames: {[p.name for p in paths]} | "
                    f"Attempt: {attempt}/{max_retries} | Duration: {duration}s"
                )
            except Exception as e:
                duration = round(time.time() - start_t, 2)
                logger.warning(
                    f"[VLM Telemetry] REQUEST_FAIL | Model: {self.model_name} | Frames: {[p.name for p in paths]} | "
                    f"Attempt: {attempt}/{max_retries} | Duration: {duration}s | Error: {e}"
                )

            if attempt < max_retries:
                time.sleep(0.5 * attempt)

        if settings.VLM_MOCK_MODE or settings.VLM_PROVIDER == "mock":
            logger.warning("[VLM Telemetry] Retries exhausted. Explicit Mock Mode enabled -> returning MockVLM.")
            return MockVLMProvider().analyze_image(paths[0], prompt)

        logger.error(f"[VLM Telemetry] FAILED_ALL_RETRIES | Frames: {[p.name for p in paths]} | Frame marked unavailable.")
        return {}


class MockVLMProvider(VLMProvider):
    """Fallback Mock VLM provider for explicit development/testing mode ONLY."""

    def analyze_image(self, image_path: Union[str, Path], prompt: str) -> Dict[str, Any]:
        path = Path(image_path)
        logger.info(f"[Mock VLM] Explicit Mock Mode active for frame {path.name}")
        return {
            "environment": "Indoor workplace / study area",
            "people": [
                {
                    "temporary_id": "Person #1",
                    "description": "Person sitting near desk wearing casual clothes",
                    "activity": "working at desk",
                    "confidence": 0.92,
                }
            ],
            "objects": [
                {
                    "name": "laptop",
                    "description": "open silver laptop",
                    "location": "on table",
                    "confidence": 0.96,
                },
                {
                    "name": "backpack",
                    "description": "black backpack",
                    "location": "beside chair",
                    "confidence": 0.88,
                },
            ],
            "activities": ["working", "gesturing", "looking at screen"],
            "interactions": ["Person #1 using laptop"],
            "relationships": ["laptop on desk", "Person #1 beside desk"],
            "visible_text": [],
            "observations": [f"Clear visual frame from {path.name}"],
            "uncertainties": ["Exact brand of backpack is unclear"],
        }


def get_vlm_provider(provider_type: str = "") -> VLMProvider:
    """Factory function returning configured VLM provider instance."""
    provider_name = (provider_type or settings.VLM_PROVIDER).lower()
    if settings.VLM_MOCK_MODE or provider_name == "mock":
        return MockVLMProvider()
    elif provider_name == "gemini":
        return GeminiVLMProvider()
    elif provider_name in ("openai", "custom"):
        return OpenAIVLMProvider()
    else:
        logger.warning(f"Unknown VLM provider '{provider_name}'. Defaulting to GeminiVLMProvider.")
        return GeminiVLMProvider()
