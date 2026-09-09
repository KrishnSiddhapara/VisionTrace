from pathlib import Path
from typing import List, Union, Dict, Any, Tuple
import cv2
import numpy as np

from config.settings import settings
from models.schemas import Scene, SampledFrame
from video.movement_detector import movement_detector
from video.frame_provider import video_frame_provider
from vision.quality import frame_quality_checker
from utils.logger import logger

class AdaptiveFrameSampler:
    """
    OpenCV Change-Driven Movement & Visual Keyframe Sampler.
    Executes frame-by-frame movement analysis using centralized VideoFrameProvider,
    creates movement event windows (start, peak, end), applies perceptual similarity filtering,
    and enforces VLM frame budget limits.
    """

    def is_visually_similar(self, frame_a: np.ndarray, frame_b: np.ndarray) -> bool:
        """
        Check if two frames are visually similar using perceptual dhash & grayscale difference.
        Returns True if frames are redundant duplicates.
        """
        if frame_a is None or frame_b is None or frame_a.shape != frame_b.shape:
            return False

        hash_a = frame_quality_checker.compute_content_hash(frame_a)
        hash_b = frame_quality_checker.compute_content_hash(frame_b)

        # Hamming distance between dhash strings
        try:
            val_a = int(hash_a, 16)
            val_b = int(hash_b, 16)
            hamming_dist = bin(val_a ^ val_b).count('1')
            if hamming_dist <= 6:  # 6 bits or fewer difference out of 64 bits = highly similar
                return True
        except Exception:
            pass

        # Fallback mean absolute pixel difference
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        diff = float(np.mean(cv2.absdiff(gray_a, gray_b)) / 255.0)
        return diff < 0.04

    def sample_scene_frames(
        self,
        video_path: Union[str, Path],
        scenes: List[Scene],
        output_dir: Union[str, Path],
        sampling_mode: str = "Balanced"
    ) -> List[SampledFrame]:
        """
        Extract change-driven representative frames where OpenCV detects meaningful visual movement or state changes.
        """
        path = Path(video_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        video_frame_provider.set_video(path)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.error(f"Could not open video for sampling: {path.name}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()

        # Step dynamically based on profile mode
        mode = sampling_mode.lower()
        if mode == "fast":
            frame_step = max(1, int(fps * 0.5))  # Every ~0.5s
            min_gap_sec = 0.6
            vlm_budget = settings.VLM_MAX_FRAMES_FAST
        elif mode == "deep analysis":
            frame_step = max(1, int(fps * 0.2))  # Every ~0.2s
            min_gap_sec = 0.25
            vlm_budget = settings.VLM_MAX_FRAMES_DEEP
        else:  # Balanced
            frame_step = max(1, int(fps * 0.3))  # Every ~0.3s
            min_gap_sec = 0.40
            vlm_budget = settings.VLM_MAX_FRAMES_BALANCED

        # Step 1: Scan video using VideoFrameProvider
        candidate_frames_meta: List[Dict[str, Any]] = []
        prev_gray_blur = None

        for frame_idx, frame in video_frame_provider.iterate_frames(step=frame_step):
            # Downscale large frames for fast OpenCV motion evaluation
            small_frame = video_frame_provider.resize_for_inference(frame, settings.INFERENCE_MAX_WIDTH, settings.INFERENCE_MAX_HEIGHT)
            motion_res = movement_detector.analyze_frame_motion(
                curr_frame=small_frame,
                prev_gray_blur=prev_gray_blur
            )
            prev_gray_blur = motion_res.get("curr_gray_blur")

            ts = round(frame_idx / fps, 2)
            candidate_frames_meta.append({
                "frame_index": frame_idx,
                "timestamp": ts,
                "motion_score": motion_res["motion_score"],
                "change_score": motion_res["change_score"],
                "motion_area_ratio": motion_res["motion_area_ratio"],
                "is_meaningful_change": motion_res["is_meaningful_change"],
                "is_camera_motion": motion_res["is_camera_motion"],
                "motion_type": motion_res["motion_type"],
            })

        # Step 2: Form Movement Event Windows and Select Keyframes
        meaningful_candidates = [meta for meta in candidate_frames_meta if meta["is_meaningful_change"]]
        selected_meta_indices: List[Tuple[Dict[str, Any], str]] = []  # (meta, selection_reason)

        if not meaningful_candidates:
            logger.info("No significant movement detected in video. Retaining initial frame for scene understanding.")
            if candidate_frames_meta:
                selected_meta_indices.append((candidate_frames_meta[0], "scene_initial_static"))
        else:
            windows: List[List[Dict[str, Any]]] = []
            curr_window: List[Dict[str, Any]] = []

            for meta in candidate_frames_meta:
                if meta["is_meaningful_change"]:
                    curr_window.append(meta)
                else:
                    if curr_window:
                        windows.append(curr_window)
                        curr_window = []

            if curr_window:
                windows.append(curr_window)

            for win in windows:
                if not win:
                    continue
                start_meta = win[0]
                peak_meta = max(win, key=lambda m: m["motion_score"])
                end_meta = win[-1]

                selected_meta_indices.append((start_meta, "movement_start"))

                if peak_meta["frame_index"] != start_meta["frame_index"] and peak_meta["frame_index"] != end_meta["frame_index"]:
                    selected_meta_indices.append((peak_meta, "movement_peak"))

                if end_meta["frame_index"] != start_meta["frame_index"]:
                    selected_meta_indices.append((end_meta, "movement_end"))

        # Add scene boundary anchors
        scene_start_frames = {s.start_frame for s in scenes}
        for meta in candidate_frames_meta:
            if meta["frame_index"] in scene_start_frames:
                if not any(sm[0]["frame_index"] == meta["frame_index"] for sm in selected_meta_indices):
                    selected_meta_indices.append((meta, "scene_boundary_anchor"))

        selected_meta_indices.sort(key=lambda item: item[0]["timestamp"])

        # Step 3: Extract Selected Keyframes using VideoFrameProvider Batch Lookup
        indices_to_fetch = [meta["frame_index"] for meta, _ in selected_meta_indices]
        fetched_frames = video_frame_provider.get_frames_batch(indices_to_fetch)

        sampled_frames: List[SampledFrame] = []
        last_kept_frame = None
        last_kept_ts = -10.0
        global_frame_counter = 1

        for meta, reason in selected_meta_indices:
            f_idx = meta["frame_index"]
            ts = meta["timestamp"]

            if (ts - last_kept_ts) < min_gap_sec and reason not in ("scene_boundary_anchor", "movement_start"):
                continue

            frame = fetched_frames.get(f_idx)
            if frame is None:
                continue

            # Perceptual similarity check
            if last_kept_frame is not None and reason not in ("scene_boundary_anchor", "movement_start"):
                if self.is_visually_similar(last_kept_frame, frame):
                    continue

            # Save frame to disk if not already present
            frame_filename = f"change_frame_{f_idx:06d}.jpg"
            save_path = out_dir / frame_filename
            if not save_path.exists() or save_path.stat().st_size == 0:
                cv2.imwrite(str(save_path), frame)

            last_kept_frame = frame.copy()
            last_kept_ts = ts

            quality_info = frame_quality_checker.evaluate_quality(save_path)

            sf = SampledFrame(
                frame_id=f"frame_{global_frame_counter:04d}",
                timestamp=ts,
                frame_index=f_idx,
                path=str(save_path.resolve()),
                scene_id=1,
                sampling_reason=reason,
                quality_score=quality_info.get("quality_score", 1.0),
                is_blurry=quality_info.get("is_blurry", False),
                content_hash=quality_info.get("content_hash", ""),
                motion_score=meta["motion_score"],
                change_score=meta["change_score"],
                motion_area=meta["motion_area_ratio"],
                selection_reason=reason,
            )
            sampled_frames.append(sf)
            global_frame_counter += 1

            if len(sampled_frames) >= vlm_budget:
                logger.info(f"[Frame Sampler] Reached VLM budget limit ({vlm_budget} frames) for profile '{sampling_mode}'.")
                break

        sampled_frames.sort(key=lambda f: f.timestamp)
        logger.info(f"OpenCV Adaptive Sampler selected {len(sampled_frames)} representative keyframes out of {total_frames} total frames (Profile: {sampling_mode}).")
        return sampled_frames

    def sample_event_dense_frames(
        self,
        video_path: Union[str, Path],
        event_windows: List[Tuple[float, float]],
        output_dir: Union[str, Path],
        existing_frames: List[SampledFrame]
    ) -> List[SampledFrame]:
        """
        PASS 2: Event-Focused Second Pass.
        Extracts dense sub-second frames around candidate event windows for VLM verification.
        """
        if not event_windows:
            return existing_frames

        path = Path(video_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        video_frame_provider.set_video(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return existing_frames

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        existing_timestamps = {f.timestamp for f in existing_frames}
        dense_frames = list(existing_frames)
        global_counter = len(existing_frames) + 1

        for start_t, end_t in event_windows:
            w_start = max(0.0, start_t - 1.0)
            w_end = end_t + 1.0
            
            sample_times = np.arange(w_start, w_end, 0.3)
            for st in sample_times:
                st_round = round(float(st), 2)
                if any(abs(st_round - ets) < 0.15 for ets in existing_timestamps):
                    continue

                frame_idx = int(st_round * fps)
                frame = video_frame_provider.get_frame(frame_idx)
                if frame is None:
                    continue

                frame_filename = f"pass2_dense_frame_{frame_idx:06d}.jpg"
                save_path = out_dir / frame_filename
                if not save_path.exists() or save_path.stat().st_size == 0:
                    cv2.imwrite(str(save_path), frame)

                dense_sf = SampledFrame(
                    frame_id=f"frame_p2_{global_counter:04d}",
                    timestamp=st_round,
                    frame_index=frame_idx,
                    path=str(save_path.resolve()),
                    scene_id=1,
                    sampling_reason="event_dense_pass2",
                    selection_reason="event_dense_pass2",
                )
                dense_frames.append(dense_sf)
                existing_timestamps.add(st_round)
                global_counter += 1

                max_vlm = getattr(settings, "MAX_VLM_FRAMES", 30)
                if len(dense_frames) >= max_vlm + 30:
                    break

        dense_frames.sort(key=lambda f: f.timestamp)
        logger.info(f"Pass 2 Event-Focused Dense Sampler extracted {len(dense_frames) - len(existing_frames)} additional verification frames.")
        return dense_frames

    def extract_tracking_frames(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        target_fps: float = 5.0
    ) -> List[SampledFrame]:
        """
        Extract continuous sub-sampled frames for internal multi-object tracking.
        Ensures high temporal continuity across crossings, occlusions, and fast motion.
        """
        path = Path(video_path)
        trk_dir = Path(output_dir) / "tracking_frames"
        trk_dir.mkdir(parents=True, exist_ok=True)

        video_frame_provider.set_video(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()

        step = max(1, int(round(fps / max(0.5, target_fps))))
        tracking_frames: List[SampledFrame] = []
        counter = 1

        for frame_idx, frame in video_frame_provider.iterate_frames(step=step):
            ts = round(frame_idx / fps, 2)
            frame_filename = f"trk_frame_{frame_idx:06d}.jpg"
            save_path = trk_dir / frame_filename

            if not save_path.exists() or save_path.stat().st_size == 0:
                cv2.imwrite(str(save_path), frame)

            sf = SampledFrame(
                frame_id=f"trk_frame_{counter:04d}",
                timestamp=ts,
                frame_index=frame_idx,
                path=str(save_path.resolve()),
                scene_id=1,
                sampling_reason="continuous_tracking",
                selection_reason="continuous_tracking",
            )
            tracking_frames.append(sf)
            counter += 1

        logger.info(f"[Tracking Frame Sampler] Extracted {len(tracking_frames)} continuous tracking frames @ ~{target_fps} FPS out of {total_frames} total frames.")
        return tracking_frames

frame_sampler = AdaptiveFrameSampler()
