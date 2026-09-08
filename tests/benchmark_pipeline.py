import time
import unittest
from pathlib import Path
import cv2
import numpy as np

from config.settings import settings
from video.processor import video_processor
from video.sampler import frame_sampler
from vision.object_detector import object_detector
from vision.vlm import MockVLMProvider
from vision.tracker import SpatialIoUTracker
from intelligence.event_detector import event_detector
from intelligence.temporal_reasoner import temporal_reasoner
from utils.profiler import profiler

class PipelinePerformanceBenchmark(unittest.TestCase):
    """
    Performance & Benchmark Test Suite.
    Measures processing duration, frame throughput, YOLO call efficiency, and VLM budget compliance.
    """

    def setUp(self):
        self.test_dir = settings.PROCESSED_DIR / "benchmark_temp"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = self.test_dir / "benchmark_vid.mp4"

        # Generate synthetic 10-second test video (30 FPS -> 300 frames)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.video_path), fourcc, 30.0, (640, 480))
        for i in range(300):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add moving circle (simulating moving object)
            x_pos = int((i * 4) % 600) + 20
            cv2.circle(frame, (x_pos, 240), 25, (0, 255, 0), -1)
            cv2.putText(frame, f"Frame {i}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            writer.write(frame)
        writer.release()

    def test_benchmark_fast_and_balanced_profiles(self):
        """Benchmark pipeline timing across Fast and Balanced profiles."""
        profiler.start_pipeline()
        
        # 1. Ingestion
        val, meta, scenes, _ = video_processor.process_video(self.video_path)
        self.assertTrue(val.is_valid)

        # 2. Fast Profile Sampling
        start_t = time.perf_counter()
        sampled_fast = frame_sampler.sample_scene_frames(self.video_path, scenes, self.test_dir / "fast_frames", sampling_mode="Fast")
        dur_fast = time.perf_counter() - start_t

        self.assertLessEqual(len(sampled_fast), settings.VLM_MAX_FRAMES_FAST)

        # 3. Balanced Profile Sampling
        start_t = time.perf_counter()
        sampled_bal = frame_sampler.sample_scene_frames(self.video_path, scenes, self.test_dir / "bal_frames", sampling_mode="Balanced")
        dur_bal = time.perf_counter() - start_t

        self.assertLessEqual(len(sampled_bal), settings.VLM_MAX_FRAMES_BALANCED)

        # 4. Batched YOLO Inference
        start_t = time.perf_counter()
        paths = [sf.path for sf in sampled_bal]
        dets = object_detector.detect_objects_batch(paths, video_hash=meta.video_hash, confidence_threshold=0.45)
        dur_yolo = time.perf_counter() - start_t

        total_dur = profiler.stop_pipeline()

        print(f"\n[BENCHMARK REPORT]")
        print(f"Total Video Frames: {meta.frame_count} ({meta.duration_sec}s)")
        print(f"Fast Profile Frames Selected: {len(sampled_fast)} (Sampling Duration: {dur_fast:.3f}s)")
        print(f"Balanced Profile Frames Selected: {len(sampled_bal)} (Sampling Duration: {dur_bal:.3f}s)")
        print(f"YOLO Batch Detection Duration: {dur_yolo:.3f}s across {len(paths)} frames")
        print(f"Total Benchmark Pipeline Duration: {total_dur:.3f}s")

if __name__ == "__main__":
    unittest.main()
