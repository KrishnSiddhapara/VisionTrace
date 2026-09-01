from pathlib import Path
from typing import List, Union, Dict, Any, Tuple
import cv2
import numpy as np

from config.settings import settings
from models.schemas import Scene, SampledFrame
from video.movement_detector import movement_detector
from vision.quality import frame_quality_checker
from utils.logger import logger

class AdaptiveFrameSampler:
    """
    OpenCV Change-Driven Movement & Visual Keyframe Sampler.
    Executes frame-by-frame movement analysis, creates movement event windows (start, peak, end),
    applies perceptual similarity filtering to remove redundant frames, and outputs representative keyframes for VLM ingestion.
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

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.error(f"Could not open video for sampling: {path.name}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()

        # Step dynamically based on mode (evaluate candidate motion every N frames)
        mode = sampling_mode.lower()
        if mode == "fast":
            frame_step = max(1, int(fps * 0.5))  # Every ~0.5s
            min_gap_sec = 0.6
        elif mode == "deep analysis":
            frame_step = max(1, int(fps * 0.2))  # Every ~0.2s
            min_gap_sec = 0.25
        else:  # Balanced
            frame_step = max(1, int(fps * 0.3))  # Every ~0.3s
            min_gap_sec = 0.40

        # Step 1: Scan video to evaluate motion scores and detect movement candidate frames
        cap = cv2.VideoCapture(str(path))
        candidate_frames_meta: List[Dict[str, Any]] = []
        prev_gray_blur = None

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % frame_step == 0 or frame_idx == 0 or frame_idx == total_frames - 1:
                motion_res = movement_detector.analyze_frame_motion(
                    curr_frame=frame,
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

            frame_idx += 1
        cap.release()

        # Step 2: Form Movement Event Windows and Select Keyframes
        # Group contiguous movement candidates into movement windows
        meaningful_candidates = [meta for meta in candidate_frames_meta if meta["is_meaningful_change"]]
        
        selected_meta_indices: List[Tuple[Dict[str, Any], str]] = []  # (meta, selection_reason)

        if not meaningful_candidates:
            logger.info("No significant movement detected in video. Retaining initial frame for scene understanding.")
            if candidate_frames_meta:
                selected_meta_indices.append((candidate_frames_meta[0], "scene_initial_static"))
        else:
            # Build movement windows
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

            # For each window, select START, PEAK, END keyframes
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

        # Add scene boundary anchors if missing
        scene_start_frames = {s.start_frame for s in scenes}
        for meta in candidate_frames_meta:
            if meta["frame_index"] in scene_start_frames:
                if not any(sm[0]["frame_index"] == meta["frame_index"] for sm in selected_meta_indices):
                    selected_meta_indices.append((meta, "scene_boundary_anchor"))

        # Sort selected frames chronologically
        selected_meta_indices.sort(key=lambda item: item[0]["timestamp"])

        # Step 3: Perceptual Similarity Filter & Frame Extraction
        cap = cv2.VideoCapture(str(path))
        sampled_frames: List[SampledFrame] = []
        last_kept_frame = None
        last_kept_ts = -10.0
        global_frame_counter = 1

        for meta, reason in selected_meta_indices:
            f_idx = meta["frame_index"]
            ts = meta["timestamp"]

            # Check min frame gap
            if (ts - last_kept_ts) < min_gap_sec and reason not in ("scene_boundary_anchor", "movement_start"):
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Perceptual similarity check against last kept frame
            if last_kept_frame is not None and reason not in ("scene_boundary_anchor", "movement_start"):
                if self.is_visually_similar(last_kept_frame, frame):
                    continue

            # Save selected change frame to disk
            frame_filename = f"change_frame_{f_idx:06d}.jpg"
            save_path = out_dir / frame_filename
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

            if len(sampled_frames) >= settings.MAX_VLM_FRAMES:
                logger.warning(f"Reached max VLM frame limit ({settings.MAX_VLM_FRAMES} frames).")
                break

        cap.release()

        sampled_frames.sort(key=lambda f: f.timestamp)
        logger.info(f"OpenCV Change Sampler selected {len(sampled_frames)} representative movement keyframes out of {total_frames} total video frames.")
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

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return existing_frames

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                frame_filename = f"pass2_dense_frame_{frame_idx:06d}.jpg"
                save_path = out_dir / frame_filename
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

                if len(dense_frames) >= settings.MAX_VLM_FRAMES + 30:
                    break

        cap.release()
        dense_frames.sort(key=lambda f: f.timestamp)
        logger.info(f"Pass 2 Event-Focused Dense Sampler extracted {len(dense_frames) - len(existing_frames)} additional verification frames.")
        return dense_frames

frame_sampler = AdaptiveFrameSampler()
