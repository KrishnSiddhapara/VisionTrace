from pathlib import Path
from typing import Union, List
import cv2

from config.settings import settings
from utils.logger import logger

class FrameExtractor:
    """Extracts frame images from video at specific timestamps or indices."""

    def extract_frame_at_index(self, video_path: Union[str, Path], frame_index: int, output_path: Union[str, Path]) -> bool:
        path = Path(video_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            cv2.imwrite(str(out_path), frame)
            return True
        return False

    def extract_frames_at_indices(self, video_path: Union[str, Path], frame_indices: List[int], output_dir: Union[str, Path]) -> List[Path]:
        path = Path(video_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        extracted_paths: List[Path] = []
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return extracted_paths

        sorted_indices = sorted(list(set(frame_indices)))
        for idx in sorted_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                out_path = out_dir / f"frame_{idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                extracted_paths.append(out_path)

        cap.release()
        return extracted_paths

frame_extractor = FrameExtractor()
