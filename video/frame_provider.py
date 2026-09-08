from pathlib import Path
from typing import Union, Optional, Tuple, Dict, List, Generator
import cv2
import numpy as np
from config.settings import settings
from utils.logger import logger

class VideoFrameProvider:
    """
    Centralized, efficient video frame access provider.
    Avoids repeatedly opening cv2.VideoCapture across multiple modules.
    Provides cached frame retrieval, single-pass video iteration, and optional inference downscaling.
    """

    def __init__(self, video_path: Optional[Union[str, Path]] = None):
        self.video_path: Optional[Path] = Path(video_path) if video_path else None
        self._frame_cache: Dict[int, np.ndarray] = {}
        self._max_cache_size: int = 150  # Prevent RAM overflow

    def set_video(self, video_path: Union[str, Path]) -> None:
        new_path = Path(video_path)
        if self.video_path != new_path:
            self.video_path = new_path
            self.clear_cache()

    def clear_cache(self) -> None:
        self._frame_cache.clear()

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Retrieve frame by index with caching."""
        if not self.video_path or not self.video_path.exists():
            return None

        if frame_index in self._frame_cache:
            return self._frame_cache[frame_index].copy()

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            if len(self._frame_cache) < self._max_cache_size:
                self._frame_cache[frame_index] = frame.copy()
            return frame
        return None

    def get_frames_batch(self, frame_indices: List[int]) -> Dict[int, np.ndarray]:
        """Fetch multiple frames efficiently in ascending index order."""
        if not self.video_path or not self.video_path.exists() or not frame_indices:
            return {}

        results: Dict[int, np.ndarray] = {}
        sorted_indices = sorted(set(frame_indices))

        # Check cached first
        missing_indices = []
        for idx in sorted_indices:
            if idx in self._frame_cache:
                results[idx] = self._frame_cache[idx].copy()
            else:
                missing_indices.append(idx)

        if not missing_indices:
            return results

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return results

        for idx in missing_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                results[idx] = frame.copy()
                if len(self._frame_cache) < self._max_cache_size:
                    self._frame_cache[idx] = frame.copy()

        cap.release()
        return results

    def iterate_frames(self, step: int = 1) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Single-pass generator for reading video frames sequentially."""
        if not self.video_path or not self.video_path.exists():
            return

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % step == 0:
                yield (frame_idx, frame)

            frame_idx += 1

        cap.release()

    def resize_for_inference(self, frame: np.ndarray, max_w: int = 1280, max_h: int = 720) -> np.ndarray:
        """Optimally downscale large frames (e.g., 4K) for fast OpenCV & YOLO processing."""
        if frame is None:
            return frame
        h, w = frame.shape[:2]
        if w <= max_w and h <= max_h:
            return frame

        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

video_frame_provider = VideoFrameProvider()
