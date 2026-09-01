from pathlib import Path
from typing import Union
import cv2

from models.schemas import VideoMetadata
from utils.file_utils import get_file_hash, get_file_size_mb
from utils.logger import logger

class VideoMetadataExtractor:
    """Extracts detailed metadata from video files using OpenCV."""

    def extract(self, video_path: Union[str, Path]) -> VideoMetadata:
        path = Path(video_path)
        size_mb = get_file_size_mb(path)
        video_hash = get_file_hash(path)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {path.name}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
        if not codec or not codec.isprintable():
            codec = path.suffix.replace(".", "").upper()

        cap.release()

        duration_sec = frame_count / fps if fps > 0 else 0.0

        metadata = VideoMetadata(
            filename=path.name,
            filepath=str(path.resolve()),
            file_size_mb=round(size_mb, 2),
            duration_sec=round(duration_sec, 2),
            fps=round(fps, 2),
            width=width,
            height=height,
            frame_count=frame_count,
            codec=codec,
            video_hash=video_hash,
            resolution_str=f"{width}x{height}",
        )

        logger.info(f"Extracted metadata: {metadata.filename} | Duration: {metadata.duration_sec}s | FPS: {metadata.fps}")
        return metadata

metadata_extractor = VideoMetadataExtractor()
