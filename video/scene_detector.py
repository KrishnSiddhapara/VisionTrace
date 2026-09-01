from pathlib import Path
from typing import List, Union
import cv2

from models.schemas import Scene
from utils.logger import logger

class SceneDetector:
    """
    Detects scene boundaries in videos using PySceneDetect when available,
    falling back to OpenCV HSV color histogram / content change analysis.
    """

    def detect_scenes(self, video_path: Union[str, Path], min_scene_len_sec: float = 1.5) -> List[Scene]:
        path = Path(video_path)
        scenes: List[Scene] = []

        try:
            from scenedetect import detect, ContentDetector
            scene_list = detect(str(path), ContentDetector())
            logger.info(f"PySceneDetect found {len(scene_list)} scenes in {path.name}")

            for idx, (start_time, end_time) in enumerate(scene_list, start=1):
                try:
                    start_sec = float(start_time.seconds)
                    end_sec = float(end_time.seconds)
                    start_frame = int(start_time.frame_num)
                    end_frame = int(end_time.frame_num)
                except AttributeError:
                    start_sec = float(start_time.get_seconds())
                    end_sec = float(end_time.get_seconds())
                    start_frame = int(start_time.get_frames())
                    end_frame = int(end_time.get_frames())

                duration = end_sec - start_sec

                scenes.append(
                    Scene(
                        scene_id=idx,
                        start_time=round(start_sec, 2),
                        end_time=round(end_sec, 2),
                        duration=round(duration, 2),
                        start_frame=start_frame,
                        end_frame=end_frame,
                    )
                )
        except Exception as e:
            logger.warning(f"PySceneDetect fallback to OpenCV content detection: {e}")
            scenes = self._fallback_opencv_detect(path, min_scene_len_sec=min_scene_len_sec)

        if not scenes:
            # Single scene fallback
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 30
            cap.release()
            duration = frame_count / fps
            scenes = [
                Scene(
                    scene_id=1,
                    start_time=0.0,
                    end_time=round(duration, 2),
                    duration=round(duration, 2),
                    start_frame=0,
                    end_frame=frame_count,
                )
            ]

        return scenes

    def _fallback_opencv_detect(self, path: Path, min_scene_len_sec: float = 1.5) -> List[Scene]:
        """OpenCV histogram diff scene boundary detection fallback."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        min_frames = int(min_scene_len_sec * fps)

        prev_hist = None
        scene_starts = [0]
        current_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame % 5 == 0:  # Sample every 5th frame for speed
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [18, 25], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

                if prev_hist is not None:
                    comp = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    if comp < 0.60 and (current_frame - scene_starts[-1]) >= min_frames:
                        scene_starts.append(current_frame)
                prev_hist = hist
            current_frame += 1

        cap.release()

        scenes = []
        for idx in range(len(scene_starts)):
            start_f = scene_starts[idx]
            end_f = scene_starts[idx + 1] if idx + 1 < len(scene_starts) else frame_count
            start_t = start_f / fps
            end_t = end_f / fps
            scenes.append(
                Scene(
                    scene_id=idx + 1,
                    start_time=round(start_t, 2),
                    end_time=round(end_t, 2),
                    duration=round(end_t - start_t, 2),
                    start_frame=start_f,
                    end_frame=end_f,
                )
            )

        return scenes

scene_detector = SceneDetector()
