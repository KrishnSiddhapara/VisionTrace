from pathlib import Path
from typing import Union, Tuple, List
from config.settings import settings
from models.schemas import VideoMetadata, Scene, SampledFrame, ValidationResult
from video.validator import video_validator
from video.metadata import metadata_extractor
from video.scene_detector import scene_detector
from video.sampler import frame_sampler
from utils.logger import logger
from utils.caching import cache_manager

class VideoProcessor:
    """Master video processor orchestrating video ingestion, scene detection, and sampling."""

    def process_video(self, video_path: Union[str, Path]) -> Tuple[ValidationResult, VideoMetadata, List[Scene], List[SampledFrame]]:
        path = Path(video_path)

        # 1. Validation
        validation = video_validator.validate(path)
        if not validation.is_valid:
            return validation, None, [], []

        # 2. Metadata
        metadata = metadata_extractor.extract(path)

        # Cache check for scenes and frames
        cache_key = cache_manager.build_versioned_key(
            prefix="video_ingest",
            video_hash=metadata.video_hash,
        )
        cached_data = cache_manager.get(cache_key)
        if cached_data:
            logger.info("Reusing cached scenes and sampled frames.")
            scenes = [Scene(**s) for s in cached_data.get("scenes", [])]
            sampled_frames = [SampledFrame(**f) for f in cached_data.get("frames", [])]
            return validation, metadata, scenes, sampled_frames

        # 3. Scene Detection
        logger.info(f"Detecting scenes for {metadata.filename}...")
        scenes = scene_detector.detect_scenes(path)

        # 4. Adaptive Sampling
        output_frames_dir = settings.PROCESSED_DIR / metadata.video_hash / "frames"
        logger.info(f"Extracting adaptive frames into {output_frames_dir}...")
        sampled_frames = frame_sampler.sample_scene_frames(path, scenes, output_frames_dir)

        # Cache result
        cache_manager.set(
            cache_key,
            {
                "metadata": metadata.model_dump(),
                "scenes": [s.model_dump() for s in scenes],
                "frames": [f.model_dump() for f in sampled_frames],
            }
        )

        return validation, metadata, scenes, sampled_frames

video_processor = VideoProcessor()
