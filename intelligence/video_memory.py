from pathlib import Path
from typing import Optional, Union, Dict, Any

from config.settings import settings
from models.schemas import VideoMemory, VideoMetadata, Scene, SampledFrame, FrameObservation, TrackedObject, VideoEvent
from utils.caching import cache_manager
from utils.logger import logger

class VideoMemoryManager:
    """Manages full structured video memory lifecycle."""

    def save_memory(self, memory: VideoMemory) -> Path:
        cache_key = f"video_memory_{memory.video_hash}"
        cache_manager.set(cache_key, memory.model_dump())

        file_path = settings.PROCESSED_DIR / f"memory_{memory.video_hash}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(memory.model_dump_json(indent=2))

        logger.info(f"Saved VideoMemory for video {memory.metadata.filename} ({memory.video_hash[:8]})")
        return file_path

    def load_memory(self, video_hash: str) -> Optional[VideoMemory]:
        cache_key = f"video_memory_{video_hash}"
        cached = cache_manager.get(cache_key)
        if cached:
            return VideoMemory(**cached)

        file_path = settings.PROCESSED_DIR / f"memory_{video_hash}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = f.read()
                return VideoMemory.model_validate_json(data)
            except Exception as e:
                logger.error(f"Error loading VideoMemory file {file_path}: {e}")
        return None

video_memory_manager = VideoMemoryManager()
