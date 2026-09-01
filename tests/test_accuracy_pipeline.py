import unittest
import tempfile
from pathlib import Path
import numpy as np
import cv2

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
                # Moving object
                cx = int(40 + i * 2.5)
                cy = 120
                cv2.circle(frame, (cx, cy), 25, (0, 255, 100), -1)
                out.write(frame)
            out.release()

            # 1. Scene detection
            scenes = scene_detector.detect_scenes(video_path)
            self.assertGreaterEqual(len(scenes), 1)

            # 2. Intelligent Multi-Criteria Sampling
            out_frames_dir = Path(tmp_dir) / "sampled_frames"
            sampled_frames = frame_sampler.sample_scene_frames(video_path, scenes, out_frames_dir, sampling_mode="Balanced")
            self.assertGreater(len(sampled_frames), 0)

            # Check timestamp ordering
            timestamps = [sf.timestamp for sf in sampled_frames]
            self.assertEqual(timestamps, sorted(timestamps))

            # 3. VLM Analysis with Mock
            analyzer = FrameAnalyzer(vlm_provider=MockVLMProvider())
            frame_obs_list = [analyzer.analyze_frame(sf) for sf in sampled_frames]
            self.assertEqual(len(frame_obs_list), len(sampled_frames))

            # 4. YOLO Object Detections & BBox IoU Spatial Tracking
            yolo_dets = {sf.frame_id: object_detector.detect_objects(sf.path) for sf in sampled_frames}
            tracks = object_tracker.track_entities(sampled_frames, yolo_dets, frame_obs_list)
            self.assertGreater(len(tracks), 0)

            # 5. Multi-Frame Verified Events
            events = event_detector.detect_events(scenes, frame_obs_list, tracks)
            self.assertGreater(len(events), 0)

            # Check that events have confidence scores
            for evt in events:
                self.assertGreater(evt.confidence, 0.0)
                self.assertLessEqual(evt.confidence, 1.0)

            # 6. Temporal Reasoning & Timeline Synthesis
            timeline = temporal_reasoner.synthesize_timeline(scenes, events, tracks, frame_obs_list)
            self.assertEqual(len(timeline), len(events))
            timeline_timestamps = [t["timestamp"] for t in timeline]
            self.assertEqual(timeline_timestamps, sorted(timeline_timestamps))

if __name__ == "__main__":
    unittest.main()
