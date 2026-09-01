import unittest
import tempfile
from pathlib import Path
import numpy as np
import cv2

from video.sampler import frame_sampler
from video.scene_detector import scene_detector

from video.movement_detector import movement_detector

class TestSampling(unittest.TestCase):

    def test_opencv_movement_detector(self):
        """Verify OpenCV movement detector on static vs moving frame pairs."""
        frame1 = np.full((240, 320, 3), 50, dtype=np.uint8)
        frame2 = np.full((240, 320, 3), 50, dtype=np.uint8)
        # Static pair
        res_static = movement_detector.analyze_frame_motion(frame2, prev_frame=frame1)
        self.assertFalse(res_static["is_meaningful_change"])
        self.assertLess(res_static["motion_score"], 0.10)

        # Moving object pair
        cv2.circle(frame2, (160, 120), 40, (255, 255, 255), -1)
        res_move = movement_detector.analyze_frame_motion(frame2, prev_frame=frame1)
        self.assertTrue(res_move["is_meaningful_change"])
        self.assertGreater(res_move["motion_score"], 0.20)

    def test_static_video_sampling(self):
        """Verify that a completely static video yields minimal representative frames."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "test_static_vid.mp4"
            fps = 30.0
            width, height = 320, 240
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            # Generate 90 completely static frames
            static_frame = np.full((height, width, 3), (100, 100, 100), dtype=np.uint8)
            for _ in range(90):
                out.write(static_frame)
            out.release()

            scenes = scene_detector.detect_scenes(video_path)
            out_frames_dir = Path(tmp_dir) / "extracted_frames_static"
            sampled_frames = frame_sampler.sample_scene_frames(video_path, scenes, out_frames_dir)

            # Should return 1 frame for scene understanding rather than sending dozens of redundant frames to VLM
            self.assertEqual(len(sampled_frames), 1)
            self.assertTrue(Path(sampled_frames[0].path).exists())

    def test_movement_video_sampling(self):
        """Verify change-driven keyframe selection with movement start/peak/end tagging."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "test_movement_vid.mp4"
            fps = 30.0
            width, height = 320, 240
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            # 1. 30 static frames
            # 2. 30 frames with moving square
            # 3. 30 static frames
            for i in range(90):
                frame = np.full((height, width, 3), (30, 30, 30), dtype=np.uint8)
                if 30 <= i < 60:
                    cx = int(20 + (i - 30) * 8)
                    cv2.rectangle(frame, (cx, 80), (cx + 40, 140), (0, 255, 255), -1)
                out.write(frame)
            out.release()

            scenes = scene_detector.detect_scenes(video_path)
            out_frames_dir = Path(tmp_dir) / "extracted_frames_move"
            sampled_frames = frame_sampler.sample_scene_frames(video_path, scenes, out_frames_dir)

            self.assertGreater(len(sampled_frames), 0)
            reasons = [sf.selection_reason for sf in sampled_frames]
            self.assertTrue(any("movement" in r or "scene" in r for r in reasons))

if __name__ == "__main__":
    unittest.main()
