import json
from pathlib import Path
from typing import Any, Optional
from config.settings import settings
from utils.logger import logger

class CacheManager:
    """Manages disk JSON cache for video analysis components."""

    def __init__(self, cache_dir: Path = settings.PROCESSED_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        safe_key = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in key])
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str) -> Optional[Any]:
        cache_file = self._get_cache_path(key)
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    logger.info(f"Cache HIT for key: {key}")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
                return None
        return None

    def set(self, key: str, value: Any) -> None:
        cache_file = self._get_cache_path(key)
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2, default=str)
            logger.info(f"Cache SET for key: {key}")
        except Exception as e:
            logger.error(f"Failed to write cache for key {key}: {e}")

    def has(self, key: str) -> bool:
        return self._get_cache_path(key).exists()

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.json"):
            try:
                f.unlink()
            except Exception as e:
                logger.warning(f"Could not delete cache file {f}: {e}")

cache_manager = CacheManager()
