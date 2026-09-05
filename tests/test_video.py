import unittest
import tempfile
from pathlib import Path
import numpy as np
import cv2

from utils.time_utils import (
    seconds_to_timestamp,
    timestamp_to_seconds,
    frame_to_timestamp,
    timestamp_to_frame,
)
from utils.file_utils import validate_file_extension, get_file_size_mb
from video.validator import video_validator
from video.metadata import metadata_extractor

class TestVideoProcessing(unittest.TestCase):

    def test_time_conversions(self):
        self.assertEqual(seconds_to_timestamp(83.42), "01:23.42")
        self.assertEqual(seconds_to_timestamp(0), "00:00.00")
        self.assertAlmostEqual(timestamp_to_seconds("01:23.42"), 83.42, delta=0.01)
        self.assertAlmostEqual(timestamp_to_seconds("00:04.00"), 4.0, delta=0.01)

        # Frame conversions @ 30 FPS
        self.assertEqual(frame_to_timestamp(120, 30.0), "00:04.00")
        self.assertEqual(timestamp_to_frame("00:04.00", 30.0), 120)

    def test_file_extension_validation(self):
        self.assertTrue(validate_file_extension("test.mp4"))
        self.assertTrue(validate_file_extension("video.mov"))
        self.assertFalse(validate_file_extension("audio.mp3"))
        self.assertFalse(validate_file_extension("doc.pdf"))

    def test_synthetic_video_processing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "test_synthetic.mp4"
            fps = 30.0
            width, height = 320, 240
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            # Generate 60 frames (2 seconds)
            for i in range(60):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.circle(frame, (50 + i * 2, 120), 20, (0, 255, 0), -1)
                out.write(frame)
            out.release()

            # 1. Validation test
            validation = video_validator.validate(video_path)
            self.assertTrue(validation.is_valid, f"Validation failed: {validation.error_message}")

            # 2. Metadata Extraction test
            meta = metadata_extractor.extract(video_path)
            self.assertEqual(meta.filename, "test_synthetic.mp4")
            self.assertEqual(meta.width, 320)
            self.assertEqual(meta.height, 240)
            self.assertAlmostEqual(meta.fps, 30.0, delta=0.5)
            self.assertEqual(meta.frame_count, 60)
            self.assertAlmostEqual(meta.duration_sec, 2.0, delta=0.1)
            self.assertNotEqual(meta.video_hash, "")

            # 3. Path sanitization test for moved directory
            from models.schemas import VideoMemory, SampledFrame
            from intelligence.video_memory import video_memory_manager
            from config.settings import settings

            stale_filepath = r"C:\Users\old_user\Desktop\visiontrace ai\uploads\test_synthetic.mp4"
            stale_sf_path = r"C:\Users\old_user\Desktop\visiontrace ai\processed\hash123\frames\test_frame.jpg"

            # Create actual uploaded file in UPLOADS_DIR
            actual_upload = settings.UPLOADS_DIR / "test_synthetic.mp4"
            actual_upload.write_bytes(video_path.read_bytes())

            meta.filepath = stale_filepath
            sf = SampledFrame(
                frame_id="frame_0001",
                timestamp=0.0,
                frame_index=0,
                path=stale_sf_path,
                scene_id=1,
            )
            mem = VideoMemory(video_hash="test_hash_sanitize", metadata=meta, sampled_frames=[sf])
            sanitized = video_memory_manager._sanitize_memory_paths(mem)

            self.assertEqual(Path(sanitized.metadata.filepath).resolve(), actual_upload.resolve())

            # Cleanup test upload file
            if actual_upload.exists():
                actual_upload.unlink()

if __name__ == "__main__":
    unittest.main()

