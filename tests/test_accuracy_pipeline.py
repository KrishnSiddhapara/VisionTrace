import unittest
import tempfile
from pathlib import Path
import numpy as np
import cv2

from PIL import Image
from config.settings import settings
from models.schemas import VideoMetadata
from video.scene_detector import scene_detector
from video.sampler import frame_sampler
from vision.object_detector import object_detector
from vision.tracker import object_tracker
from vision.vlm import MockVLMProvider
from vision.frame_analyzer import FrameAnalyzer
from intelligence.event_detector import event_detector
from intelligence.temporal_reasoner import temporal_reasoner

class TestAccuracyPipeline(unittest.TestCase):

    def test_end_to_end_accuracy_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "test_accuracy_clip.mp4"
            fps = 30.0
            width, height = 320, 240
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            # Generate 90 frames (3 seconds) with moving circle
            for i in range(90):
                frame = np.full((height, width, 3), (30, 30, 30), dtype=np.uint8)
                cx = int(40 + i * 2.5)
                cy = 120
                cv2.circle(frame, (cx, cy), 25, (0, 255, 100), -1)
                out.write(frame)
            out.release()

            # 1. Scene detection
            scenes = scene_detector.detect_scenes(video_path)
            self.assertGreaterEqual(len(scenes), 1)

            # 2. Intelligent Multi-Criteria Sampling (Pass 1)
            out_frames_dir = Path(tmp_dir) / "sampled_frames"
            sampled_frames = frame_sampler.sample_scene_frames(video_path, scenes, out_frames_dir, sampling_mode="Balanced")
            self.assertGreater(len(sampled_frames), 0)

            # Check timestamp ordering
            timestamps = [sf.timestamp for sf in sampled_frames]
            self.assertEqual(timestamps, sorted(timestamps))

            # 3. VLM Analysis with Mock
            analyzer = FrameAnalyzer(vlm_provider=MockVLMProvider())
            frame_obs_list = [analyzer.analyze_frame_window(sf) for sf in sampled_frames]
            self.assertEqual(len(frame_obs_list), len(sampled_frames))

            # 4. YOLO Object Detections & BBox IoU Spatial Tracking
            yolo_dets = {sf.frame_id: object_detector.detect_objects(sf.path) for sf in sampled_frames}
            tracks = object_tracker.track_entities(sampled_frames, yolo_dets, frame_obs_list)
            self.assertGreater(len(tracks), 0)

            # 5. Multi-Frame Verified Events & Pass 2 Dense Sampling
            candidate_events = event_detector.detect_events(scenes, frame_obs_list, tracks)
            self.assertGreater(len(candidate_events), 0)

            event_windows = [(e.start_time, e.end_time) for e in candidate_events if e.event_type != "SCENE"]
            dense_frames = frame_sampler.sample_event_dense_frames(video_path, event_windows, out_frames_dir, sampled_frames)
            self.assertGreaterEqual(len(dense_frames), len(sampled_frames))

            # 6. Event Evidence Levels Verification
            for evt in candidate_events:
                self.assertIn(evt.evidence_level, ("CONFIRMED", "PROBABLE", "UNCERTAIN", "REJECTED"))
                self.assertGreater(evt.confidence, 0.0)
                self.assertLessEqual(evt.confidence, 1.0)

            # 7. Temporal Reasoning & Timeline Synthesis
            timeline = temporal_reasoner.synthesize_timeline(scenes, candidate_events, tracks, frame_obs_list)
            self.assertEqual(len(timeline), len(candidate_events))
            timeline_timestamps = [t["timestamp"] for t in timeline]
            self.assertEqual(timeline_timestamps, sorted(timeline_timestamps))

            # 8. Final Summary Synthesis (OBJECTS, PEOPLE, FINAL DESCRIPTION)
            metadata_stub = VideoMetadata(
                video_hash="hash123", filename="test.mp4", filepath=str(video_path),
                file_size_mb=1.0, duration_sec=3.0, fps=30.0, frame_count=90,
                width=320, height=240, codec="mp4v"
            )
            final_summary = temporal_reasoner.generate_final_summary(
                metadata_stub, scenes, timeline, tracks, frame_obs_list, sampled_frames
            )
            self.assertIsNotNone(final_summary)
            self.assertIsInstance(final_summary.objects, list)
            self.assertIsInstance(final_summary.people, list)
            self.assertIsInstance(final_summary.final_description, str)

    def test_vlm_no_mock_failure_handling(self):
        """Verify that when VLM_MOCK_MODE=False and client is None, VLM returns empty response rather than fake data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "test.jpg"
            img = Image.new("RGB", (100, 100), color="red")
            img.save(img_path)

            from vision.vlm import GeminiVLMProvider
            provider = GeminiVLMProvider()
            provider.client = None  # Force uninitialized client
            
            # Ensure VLM_MOCK_MODE is False
            original_mock_setting = settings.VLM_MOCK_MODE
            settings.VLM_MOCK_MODE = False
            try:
                res = provider.analyze_image(img_path, "Analyze this image")
                self.assertEqual(res, {})  # Must return empty dict, NOT mock observations!
            finally:
                settings.VLM_MOCK_MODE = original_mock_setting

    def test_fresh_analysis_same_video_twice(self):
        """Verify that analyzing the same video twice generates unique analysis hashes without cache hits."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "test_duplicate.mp4"
            fps = 30.0
            width, height = 320, 240
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            for i in range(30):
                frame = np.full((height, width, 3), (i * 5, i * 5, i * 5), dtype=np.uint8)
                out.write(frame)
            out.release()

            from frontend.dashboard import run_full_pipeline

            mem1 = run_full_pipeline(video_path, sampling_mode="Fast")
            mem2 = run_full_pipeline(video_path, sampling_mode="Fast")

            self.assertNotEqual(mem1.video_hash, mem2.video_hash)

if __name__ == "__main__":
    unittest.main()
