import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from config.settings import settings
from utils.logger import logger

class OpenCVMovementDetector:
    """
    OpenCV Movement & Visual Change Detection Engine.
    Executes grayscale normalization, Gaussian blurring, binary thresholding,
    morphological noise reduction, contour motion area filtering, and Farneback optical flow
    camera motion estimation to identify meaningful visual movement.
    """

    def __init__(
        self,
        motion_threshold: float = None,
        min_motion_area: float = None,
        change_threshold: float = None
    ):
        self.motion_threshold = motion_threshold or settings.MOTION_THRESHOLD
        self.min_motion_area = min_motion_area or settings.MIN_MOTION_AREA
        self.change_threshold = change_threshold or settings.CHANGE_THRESHOLD

    def detect_global_camera_motion(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> Tuple[bool, float]:
        """
        Estimate global camera motion (panning, tilting, camera shake) using Farneback Optical Flow.
        Returns (is_camera_motion, mean_magnitude).
        """
        if prev_gray is None or curr_gray is None or prev_gray.shape != curr_gray.shape:
            return False, 0.0

        try:
            # Downsample for fast optical flow estimation
            h, w = prev_gray.shape
            scale = 240.0 / max(h, w)
            if scale < 1.0:
                p_small = cv2.resize(prev_gray, (int(w * scale), int(h * scale)))
                c_small = cv2.resize(curr_gray, (int(w * scale), int(h * scale)))
            else:
                p_small, c_small = prev_gray, curr_gray

            flow = cv2.calcOpticalFlowFarneback(
                p_small, c_small, None,
                pyr_scale=0.5, levels=2, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0
            )

            fx, fy = flow[..., 0], flow[..., 1]
            magnitude = np.hypot(fx, fy)
            mean_mag = float(np.mean(magnitude))
            std_mag = float(np.std(magnitude))

            # Camera motion characteristic: uniform non-zero motion vectors across >75% of frame
            is_uniform_motion = mean_mag > 1.2 and (std_mag / max(0.1, mean_mag)) < 0.65
            return is_uniform_motion, round(mean_mag, 2)
        except Exception as e:
            logger.warning(f"Error computing optical flow: {e}")
            return False, 0.0

    def analyze_frame_motion(
        self,
        curr_frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
        prev_gray_blur: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze frame for visual difference, motion mask, contour area, and camera motion.
        """
        if curr_frame is None or curr_frame.size == 0:
            return {
                "motion_score": 0.0,
                "change_score": 0.0,
                "motion_area_ratio": 0.0,
                "is_meaningful_change": False,
                "is_camera_motion": False,
                "motion_type": "static",
                "curr_gray_blur": None,
            }

        # 1. Grayscale & Gaussian Blur (5, 5) to remove noise
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)

        if prev_frame is None and prev_gray_blur is None:
            return {
                "motion_score": 1.0,
                "change_score": 1.0,
                "motion_area_ratio": 1.0,
                "is_meaningful_change": True,
                "is_camera_motion": False,
                "motion_type": "scene_initial",
                "curr_gray_blur": curr_blur,
            }

        prev_blur = prev_gray_blur
        if prev_blur is None and prev_frame is not None:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            prev_blur = cv2.GaussianBlur(prev_gray, (5, 5), 0)

        # 2. Frame Difference
        diff = cv2.absdiff(prev_blur, curr_blur)
        mean_diff = float(np.mean(diff))

        # 3. Binary Thresholding
        _, thresh = cv2.threshold(diff, int(self.motion_threshold), 255, cv2.THRESH_BINARY)

        # 4. Morphological Filtering (OPEN & CLOSE with 3x3 kernel)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        change_ratio = float(cv2.countNonZero(mask) / mask.size)

        # 5. Motion Contours & Area Filtering
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_area_sum = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.min_motion_area:
                motion_area_sum += area

        total_area = float(curr_frame.shape[0] * curr_frame.shape[1])
        motion_area_ratio = round(motion_area_sum / max(1.0, total_area), 4)

        # 6. Global Camera Motion Check
        is_camera_motion = False
        if settings.USE_OPTICAL_FLOW and prev_blur is not None:
            is_camera_motion, _ = self.detect_global_camera_motion(prev_blur, curr_blur)

        # 7. Combined Motion & Change Score
        raw_score = (
            0.40 * min(1.0, mean_diff / 40.0) +
            0.35 * min(1.0, motion_area_ratio * 5.0) +
            0.25 * min(1.0, change_ratio * 6.0)
        )

        if is_camera_motion:
            # Suppress false object motion score if camera itself is moving
            raw_score *= 0.45

        motion_score = round(float(min(1.0, max(0.0, raw_score))), 3)
        change_score = round(float(min(1.0, max(0.0, mean_diff / 50.0))), 3)

        # Determine Motion Classification
        if is_camera_motion:
            motion_type = "camera_motion"
        elif motion_score >= 0.60:
            motion_type = "major_movement"
        elif motion_score >= 0.30:
            motion_type = "meaningful_movement"
        elif motion_score >= 0.10:
            motion_type = "minor_change"
        else:
            motion_type = "static"

        is_meaningful = motion_score >= self.change_threshold or motion_area_ratio > 0.05

        return {
            "motion_score": motion_score,
            "change_score": change_score,
            "motion_area_ratio": motion_area_ratio,
            "is_meaningful_change": is_meaningful,
            "is_camera_motion": is_camera_motion,
            "motion_type": motion_type,
            "curr_gray_blur": curr_blur,
        }

movement_detector = OpenCVMovementDetector()
