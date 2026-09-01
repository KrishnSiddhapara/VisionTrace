import unittest
import tempfile
from pathlib import Path
import numpy as np
import cv2

from video.sampler import frame_sampler
from video.scene_detector import scene_detector

class TestSampling(unittest.TestCase):

    def test_adaptive_sampling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "test_sample_vid.mp4"
            fps = 30.0
            width, height = 320, 240
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            for i in range(150):
                frame = np.full((height, width, 3), (i % 255, 100, 200), dtype=np.uint8)
                out.write(frame)
            out.release()

            scenes = scene_detector.detect_scenes(video_path)
            self.assertGreaterEqual(len(scenes), 1)

            out_frames_dir = Path(tmp_dir) / "extracted_frames"
            sampled_frames = frame_sampler.sample_scene_frames(video_path, scenes, out_frames_dir)

            self.assertGreater(len(sampled_frames), 0)
            self.assertTrue(Path(sampled_frames[0].path).exists())

if __name__ == "__main__":
    unittest.main()
