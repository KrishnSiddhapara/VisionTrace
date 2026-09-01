import json
import os
import re
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


class GeminiVLMProvider(VLMProvider):
    """Google Gemini VLM Provider implementation."""

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
        if not self.client:
            logger.warning("Gemini API key missing or client uninitialized. Using mock VLM response.")
            return MockVLMProvider().analyze_image(image_path, prompt)

        path = Path(image_path)
        try:
            image = Image.open(path).convert("RGB")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[image, prompt],
            )
            raw_text = response.text
            parsed = parse_vlm_json_response(raw_text)
            if parsed:
                return parsed
            logger.warning(f"Gemini output for {path.name} could not be parsed as JSON. Using mock fallback.")
            return MockVLMProvider().analyze_image(image_path, prompt)
        except Exception as e:
            logger.error(f"Gemini VLM API error for {path.name}: {e}. Falling back to mock VLM.")
            return MockVLMProvider().analyze_image(image_path, prompt)


class OpenAIVLMProvider(VLMProvider):
    """OpenAI Vision VLM Provider implementation."""

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
        if not self.client:
            return MockVLMProvider().analyze_image(image_path, prompt)

        import base64
        path = Path(image_path)
        try:
            with open(path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            parsed = parse_vlm_json_response(content)
            return parsed if parsed else MockVLMProvider().analyze_image(image_path, prompt)
        except Exception as e:
            logger.error(f"OpenAI VLM API error for {path.name}: {e}. Falling back to mock VLM.")
            return MockVLMProvider().analyze_image(image_path, prompt)


class MockVLMProvider(VLMProvider):
    """Fallback Mock VLM provider for local testing and zero-cost simulation."""

    def analyze_image(self, image_path: Union[str, Path], prompt: str) -> Dict[str, Any]:
        path = Path(image_path)
        logger.info(f"Mock VLM analyzing frame {path.name}")
        return {
            "environment": "Indoor workplace / study area",
            "people": [
                {
                    "temporary_id": "Person #1",
                    "description": "Person sitting near desk wearing casual clothes",
                    "activity": "working at desk",
                    "location": "center left",
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
    if provider_name == "gemini":
        return GeminiVLMProvider()
    elif provider_name in ("openai", "custom"):
        return OpenAIVLMProvider()
    else:
        return MockVLMProvider()
