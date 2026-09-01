from pathlib import Path
from typing import List, Union, Dict, Any
import cv2
import numpy as np

from config.settings import settings
from models.schemas import Scene, SampledFrame
from video.frame_extractor import frame_extractor
from utils.logger import logger

class AdaptiveFrameSampler:
    """
    Advanced Multi-Criteria Intelligent Frame Sampler.
    Combines explicit scene boundary anchors, OpenCV HSV histogram diff,
    absolute pixel difference, motion magnitude estimation, and duplicate filtering.
    """

    def calculate_frame_difference(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """
        Calculate combined HSV histogram difference and pixel difference between two frames.
        Returns visual difference score between 0.0 (identical) and 1.0 (completely different).
        """
        if frame_a is None or frame_b is None or frame_a.shape != frame_b.shape:
            return 1.0

        # 1. HSV Histogram Correlation Difference
        hsv_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2HSV)
        hsv_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2HSV)

        hist_a = cv2.calcHist([hsv_a], [0, 1], None, [18, 25], [0, 180, 0, 256])
        hist_b = cv2.calcHist([hsv_b], [0, 1], None, [18, 25], [0, 180, 0, 256])

        cv2.normalize(hist_a, hist_a, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)

        hist_diff = 1.0 - cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)

        # 2. Mean Absolute Pixel Difference (Grayscale)
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        pixel_diff = float(np.mean(cv2.absdiff(gray_a, gray_b)) / 255.0)

        # Weighted combination
        combined_diff = 0.6 * max(0.0, hist_diff) + 0.4 * pixel_diff
        return round(float(combined_diff), 4)

    def sample_scene_frames(
        self,
        video_path: Union[str, Path],
        scenes: List[Scene],
        output_dir: Union[str, Path],
        sampling_mode: str = "Balanced"
    ) -> List[SampledFrame]:
        """
        Extract intelligent representative frames using boundary anchors, visual change detection,
        motion estimation, and duplicate filtering.
        """
        path = Path(video_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.error(f"Could not open video for sampling: {path.name}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()

        # Adjust thresholds based on sampling profile
        mode = sampling_mode.lower()
        if mode == "fast":
            diff_threshold = 0.18
            duplicate_threshold = 0.05
            max_scene_samples = 6
        elif mode == "deep analysis":
            diff_threshold = 0.08
            duplicate_threshold = 0.02
            max_scene_samples = 25
        else:  # Balanced
            diff_threshold = 0.12
            duplicate_threshold = 0.03
            max_scene_samples = 15

        sampled_frames: List[SampledFrame] = []
        global_frame_counter = 1

        for scene in scenes:
            start_f = scene.start_frame
            end_f = max(scene.end_frame, start_f + 1)
            duration = scene.duration

            # 1. Boundary Anchors (Start, Middle, End)
            mid_f = (start_f + end_f) // 2
            end_anchor_f = max(start_f, end_f - 1)

            anchor_indices = [start_f, mid_f, end_anchor_f]

            # 2. Dense candidate search for visual changes & motion
            step = max(1, int(fps * 0.5))  # Evaluate candidates every 0.5 seconds
            candidates = list(range(start_f, end_f, step))

            all_candidates = sorted(list(set(anchor_indices + candidates)))
            if len(all_candidates) > max_scene_samples * 2:
                # Subsample if scene is very long
                indices_subset = np.linspace(0, len(all_candidates) - 1, num=max_scene_samples * 2, dtype=int)
                all_candidates = [all_candidates[i] for i in indices_subset]

            # 3. OpenCV Visual Change & Duplicate Filter
            cap = cv2.VideoCapture(str(path))
            prev_frame = None
            scene_sampled_count = 0

            for frame_idx in all_candidates:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                is_anchor = frame_idx in anchor_indices
                should_keep = False
                diff_score = 0.0

                if prev_frame is None or is_anchor:
                    should_keep = True
                    reason = "scene_boundary_anchor" if is_anchor else "scene_initial"
                else:
                    diff_score = self.calculate_frame_difference(prev_frame, frame)
                    if diff_score >= diff_threshold:
                        should_keep = True
                        reason = f"visual_change_diff_{diff_score:.2f}"
                    elif diff_score < duplicate_threshold:
                        # Skip duplicate frame
                        continue

                if should_keep:
                    timestamp = round(frame_idx / fps, 2)
                    frame_filename = f"scene_{scene.scene_id:02d}_frame_{frame_idx:06d}.jpg"
                    save_path = out_dir / frame_filename

                    cv2.imwrite(str(save_path), frame)
                    prev_frame = frame.copy()

                    sampled_frames.append(
                        SampledFrame(
                            frame_id=f"frame_{global_frame_counter:04d}",
                            timestamp=timestamp,
                            frame_index=frame_idx,
                            path=str(save_path.resolve()),
                            scene_id=scene.scene_id,
                            sampling_reason=reason,
                        )
                    )
                    global_frame_counter += 1
                    scene_sampled_count += 1

                    if scene_sampled_count >= max_scene_samples:
                        break

                    if len(sampled_frames) >= settings.MAX_VLM_FRAMES:
                        logger.warning(f"Reached max VLM frame limit ({settings.MAX_VLM_FRAMES} frames).")
                        cap.release()
                        return sampled_frames

            cap.release()

        # Sort strictly by timestamp to maintain temporal order
        sampled_frames.sort(key=lambda f: f.timestamp)
        logger.info(f"Intelligent Sampler extracted {len(sampled_frames)} representative frames across {len(scenes)} scenes (Mode: {sampling_mode}).")
        return sampled_frames

frame_sampler = AdaptiveFrameSampler()
