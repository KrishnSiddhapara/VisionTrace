from pathlib import Path
from typing import Union
import cv2

from config.settings import settings
from models.schemas import ValidationResult
from utils.file_utils import get_file_size_mb, validate_file_extension
from utils.logger import logger

class VideoValidator:
    """Validates video files before processing."""

    def validate(self, video_path: Union[str, Path]) -> ValidationResult:
        path = Path(video_path)

        # 1. Existence check
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                error_message=f"File not found: {path.name}"
            )

        # 2. Extension check
        if not validate_file_extension(path):
            supported = ", ".join(settings.SUPPORTED_EXTENSIONS)
            return ValidationResult(
                is_valid=False,
                error_message=f"Unsupported format '{path.suffix}'. Supported formats: {supported}"
            )

        # 3. File size check
        size_mb = get_file_size_mb(path)
        if size_mb <= 0:
            return ValidationResult(
                is_valid=False,
                error_message="Video file is empty (0 MB)."
            )

        if size_mb > settings.MAX_VIDEO_SIZE_MB:
            return ValidationResult(
                is_valid=False,
                error_message=f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {settings.MAX_VIDEO_SIZE_MB} MB."
            )

        # 4. Open with OpenCV to check readability
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return ValidationResult(
                is_valid=False,
                error_message="Failed to read video file. File may be corrupted or unreadable."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            return ValidationResult(
                is_valid=False,
                error_message="Invalid video stream properties (FPS, resolution, or frame count is 0)."
            )

        duration_sec = frame_count / fps
        if duration_sec > settings.MAX_VIDEO_DURATION_SEC:
            return ValidationResult(
                is_valid=False,
                error_message=f"Video duration ({duration_sec:.1f}s) exceeds maximum allowed limit of {settings.MAX_VIDEO_DURATION_SEC}s."
            )

        logger.info(f"Video validation successful for {path.name} ({duration_sec:.1f}s, {width}x{height}, {fps:.1f} FPS)")
        return ValidationResult(is_valid=True)

video_validator = VideoValidator()
