from pathlib import Path
from typing import Optional, Union, Dict, Any

from config.settings import settings
from models.schemas import VideoMemory, VideoMetadata, Scene, SampledFrame, FrameObservation, TrackedObject, VideoEvent
from utils.caching import cache_manager
from utils.logger import logger

class VideoMemoryManager:
    """Manages full structured video memory lifecycle."""

    def _sanitize_memory_paths(self, memory: VideoMemory) -> VideoMemory:
        """Sanitize and repair file paths in VideoMemory if workspace/directory moved."""
        if memory.metadata and memory.metadata.filepath:
            v_path = Path(memory.metadata.filepath)
            if not v_path.exists():
                fallback = settings.UPLOADS_DIR / memory.metadata.filename
                if fallback.exists():
                    memory.metadata.filepath = str(fallback.resolve())

        for sf in memory.sampled_frames:
            if sf.path:
                sf_path = Path(sf.path)
                if not sf_path.exists():
                    fname = sf_path.name
                    candidates = [
                        settings.PROCESSED_DIR / memory.metadata.video_hash / "frames" / fname,
                        settings.PROCESSED_DIR / memory.metadata.video_hash / fname,
                        settings.PROCESSED_DIR / fname,
                    ]
                    for cand in candidates:
                        if cand.exists():
                            sf.path = str(cand.resolve())
                            break

        for scene in memory.scenes:
            new_kps = []
            for kp in scene.keyframe_paths:
                kp_path = Path(kp)
                if not kp_path.exists():
                    fname = kp_path.name
                    candidates = [
                        settings.PROCESSED_DIR / memory.metadata.video_hash / "frames" / fname,
                        settings.PROCESSED_DIR / memory.metadata.video_hash / fname,
                        settings.PROCESSED_DIR / fname,
                    ]
                    found = False
                    for cand in candidates:
                        if cand.exists():
                            new_kps.append(str(cand.resolve()))
                            found = True
                            break
                    if not found:
                        new_kps.append(kp)
                else:
                    new_kps.append(kp)
            scene.keyframe_paths = new_kps

        return memory

    def save_memory(self, memory: VideoMemory) -> Path:
        import json
        cache_key = f"video_memory_{memory.video_hash}"
        cache_manager.set(cache_key, memory.model_dump())

        file_path = settings.PROCESSED_DIR / f"memory_{memory.video_hash}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(memory.model_dump_json(indent=2))

        # Save structured intermediate debug files under analysis/ directory
        if memory.metadata and memory.metadata.video_hash:
            analysis_dir = settings.PROCESSED_DIR / memory.metadata.video_hash / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)

            try:
                with open(analysis_dir / "video_metadata.json", "w", encoding="utf-8") as f:
                    f.write(memory.metadata.model_dump_json(indent=2))

                with open(analysis_dir / "selected_frames.json", "w", encoding="utf-8") as f:
                    json.dump([sf.model_dump() for sf in memory.sampled_frames], f, indent=2, default=str)

                with open(analysis_dir / "vlm_observations.json", "w", encoding="utf-8") as f:
                    json.dump([fo.model_dump() for fo in memory.frame_observations], f, indent=2, default=str)

                obj_tracks = [t.model_dump() for t in memory.tracks if t.object_type.lower() != "person"]
                person_tracks = [t.model_dump() for t in memory.tracks if t.object_type.lower() == "person"]

                with open(analysis_dir / "object_tracks.json", "w", encoding="utf-8") as f:
                    json.dump(obj_tracks, f, indent=2, default=str)

                with open(analysis_dir / "person_tracks.json", "w", encoding="utf-8") as f:
                    json.dump(person_tracks, f, indent=2, default=str)

                with open(analysis_dir / "verified_events.json", "w", encoding="utf-8") as f:
                    json.dump([e.model_dump() for e in memory.events], f, indent=2, default=str)

                if memory.final_summary:
                    with open(analysis_dir / "final_summary.json", "w", encoding="utf-8") as f:
                        f.write(memory.final_summary.model_dump_json(indent=2))

                logger.info(f"Persisted intermediate diagnostic JSON files in {analysis_dir}")
            except Exception as e:
                logger.warning(f"Could not save intermediate analysis files: {e}")

        logger.info(f"Saved VideoMemory for video {memory.metadata.filename} ({memory.video_hash[:8]})")
        return file_path

    def load_memory(self, video_hash: str) -> Optional[VideoMemory]:
        if getattr(settings, "DISABLE_VIDEO_CACHE", True):
            return None

        cache_key = f"video_memory_{video_hash}"
        cached = cache_manager.get(cache_key)
        if cached:
            mem = VideoMemory(**cached)
            return self._sanitize_memory_paths(mem)

        file_path = settings.PROCESSED_DIR / f"memory_{video_hash}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = f.read()
                mem = VideoMemory.model_validate_json(data)
                return self._sanitize_memory_paths(mem)
            except Exception as e:
                logger.error(f"Error loading VideoMemory file {file_path}: {e}")
        return None

video_memory_manager = VideoMemoryManager()
