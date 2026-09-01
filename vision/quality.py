import hashlib
from pathlib import Path
from typing import Union, Dict, Any, Tuple
import cv2
import numpy as np

from utils.logger import logger

class FrameQualityChecker:
    """
    Evaluates visual quality of video frames before VLM ingestion.
    Detects blur, extreme darkness/brightness, low resolution, and computes content hashes.
    """

    def __init__(self, blur_threshold: float = 80.0):
        self.blur_threshold = blur_threshold

    def compute_content_hash(self, frame: np.ndarray) -> str:
        """Compute perceptual dhash string for frame deduplication and cache identity."""
        if frame is None:
            return "empty"
        try:
            resized = cv2.resize(frame, (9, 8), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # Difference between neighboring pixels
            diff = gray[:, 1:] > gray[:, :-1]
            decimal_value = 0
            hash_bits = diff.flatten()
            for bit in hash_bits:
                decimal_value = (decimal_value << 1) | int(bit)
            return f"{decimal_value:016x}"
        except Exception as e:
            logger.warning(f"Error computing content hash: {e}")
            return hashlib.md5(frame.tobytes()[:1000]).hexdigest()

    def evaluate_quality(self, image_input: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate frame blur (Laplacian variance), contrast, brightness, and resolution.
        Returns quality metadata dict.
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                return {"is_valid": False, "error": "File does not exist", "quality_score": 0.0, "content_hash": "empty"}
            frame = cv2.imread(str(path))
        else:
            frame = image_input

        if frame is None or frame.size == 0:
            return {"is_valid": False, "error": "Invalid frame data", "quality_score": 0.0, "content_hash": "empty"}

        h, w, c = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Blur evaluation using Laplacian variance
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = laplacian_var < self.blur_threshold

        # 2. Brightness & Contrast
        mean_brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        is_dark = mean_brightness < 25.0
        is_overexposed = mean_brightness > 230.0

        # 3. Overall quality score (0.0 to 1.0)
        quality_score = 1.0
        if is_blurry:
            quality_score -= 0.35
        if is_dark or is_overexposed:
            quality_score -= 0.30
        if w < 240 or h < 240:
            quality_score -= 0.20

        quality_score = max(0.10, round(quality_score, 2))
        content_hash = self.compute_content_hash(frame)

        return {
            "is_valid": True,
            "quality_score": quality_score,
            "laplacian_var": round(laplacian_var, 2),
            "is_blurry": is_blurry,
            "mean_brightness": round(mean_brightness, 1),
            "contrast": round(contrast, 1),
            "width": w,
            "height": h,
            "content_hash": content_hash,
        }

frame_quality_checker = FrameQualityChecker()
