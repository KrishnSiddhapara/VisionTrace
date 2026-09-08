import unittest
from models.schemas import SampledFrame, YOLODetection, FrameObservation, VideoMetadata, Scene
from vision.tracker import SpatialIoUTracker
from intelligence.temporal_reasoner import TemporalReasoner
from intelligence.event_detector import EventDetector


class TestSoccerAccuracy(unittest.TestCase):

    def test_single_moving_ball_deduplication(self):
        """
        Test soccer video scenario: 3 people playing with 1 soccer ball moving across frames.
        Verify tracker and final summary yield 3 people and EXACTLY 1 ball (no 2-ball duplication).
        """
        tracker = SpatialIoUTracker()
        reasoner = TemporalReasoner()
        event_engine = EventDetector()

        sampled_frames = [
            SampledFrame(frame_id="frame_001", timestamp=1.0, frame_index=30, path="f1.jpg", scene_id=1),
            SampledFrame(frame_id="frame_002", timestamp=3.0, frame_index=90, path="f2.jpg", scene_id=1),
            SampledFrame(frame_id="frame_003", timestamp=5.0, frame_index=150, path="f3.jpg", scene_id=1),
            SampledFrame(frame_id="frame_004", timestamp=7.0, frame_index=210, path="f4.jpg", scene_id=1),
        ]

        # YOLO detects 3 people and 1 ball in each frame (ball moving across x coordinates)
        frame_detections = {
            "frame_001": [
                YOLODetection(class_name="person", confidence=0.92, bbox=[50.0, 50.0, 100.0, 200.0]),
                YOLODetection(class_name="person", confidence=0.90, bbox=[150.0, 50.0, 200.0, 200.0]),
                YOLODetection(class_name="person", confidence=0.88, bbox=[250.0, 50.0, 300.0, 200.0]),
                YOLODetection(class_name="sports ball", confidence=0.85, bbox=[60.0, 180.0, 80.0, 200.0]),
            ],
            "frame_002": [
                YOLODetection(class_name="person", confidence=0.93, bbox=[55.0, 50.0, 105.0, 200.0]),
                YOLODetection(class_name="person", confidence=0.91, bbox=[155.0, 50.0, 205.0, 200.0]),
                YOLODetection(class_name="person", confidence=0.89, bbox=[255.0, 50.0, 305.0, 200.0]),
                # Ball moved to middle (IoU with frame 1 is 0.0, but center distance / category matches track)
                YOLODetection(class_name="sports ball", confidence=0.84, bbox=[160.0, 180.0, 180.0, 200.0]),
            ],
            "frame_003": [
                YOLODetection(class_name="person", confidence=0.91, bbox=[60.0, 50.0, 110.0, 200.0]),
                YOLODetection(class_name="person", confidence=0.92, bbox=[160.0, 50.0, 210.0, 200.0]),
                YOLODetection(class_name="person", confidence=0.90, bbox=[260.0, 50.0, 310.0, 200.0]),
                # Ball passed to third person
                YOLODetection(class_name="sports ball", confidence=0.86, bbox=[245.0, 180.0, 265.0, 200.0]),
            ],
            "frame_004": [
                YOLODetection(class_name="person", confidence=0.94, bbox=[65.0, 50.0, 115.0, 200.0]),
                YOLODetection(class_name="person", confidence=0.93, bbox=[165.0, 50.0, 215.0, 200.0]),
                YOLODetection(class_name="person", confidence=0.91, bbox=[265.0, 50.0, 315.0, 200.0]),
                YOLODetection(class_name="sports ball", confidence=0.87, bbox=[250.0, 180.0, 270.0, 200.0]),
            ],
        }

        frame_observations = [
            FrameObservation(frame_id="frame_001", timestamp=1.0, scene_id=1, environment="Soccer field", is_analyzed=True),
            FrameObservation(frame_id="frame_002", timestamp=3.0, scene_id=1, environment="Soccer field", is_analyzed=True),
            FrameObservation(frame_id="frame_003", timestamp=5.0, scene_id=1, environment="Soccer field", is_analyzed=True),
            FrameObservation(frame_id="frame_004", timestamp=7.0, scene_id=1, environment="Soccer field", is_analyzed=True),
        ]

        # Run Tracker
        tracks = tracker.track_entities(sampled_frames, frame_detections, frame_observations)

        # Count person and ball tracks
        person_tracks = [t for t in tracks if t.canonical_name == "person" or t.object_type == "person"]
        ball_tracks = [t for t in tracks if t.canonical_name == "sports ball" or t.object_type == "sports ball"]

        self.assertEqual(len(person_tracks), 3, f"Expected 3 distinct person tracks, got {len(person_tracks)}")
        self.assertEqual(len(ball_tracks), 1, f"Expected EXACTLY 1 ball track, got {len(ball_tracks)}")

        # Verify Final Summary Generation
        scenes = [Scene(scene_id=1, start_time=0.0, end_time=8.0, duration=8.0, start_frame=0, end_frame=240)]
        events = event_engine.detect_events(scenes, frame_observations, tracks)
        timeline = reasoner.synthesize_timeline(scenes, events, tracks, frame_observations)
        metadata = VideoMetadata(
            filename="soccer.mp4", filepath="soccer.mp4", file_size_mb=2.0, duration_sec=8.0,
            fps=30.0, width=640, height=480, frame_count=240, codec="h264", video_hash="soccer123"
        )
        final_summary = reasoner.generate_final_summary(
            metadata, scenes, timeline, tracks, frame_observations, sampled_frames
        )

        # Assert final object count is 1 ball and 3 people
        self.assertEqual(len(final_summary.objects), 1, f"Expected 1 ball object in FinalSummary, got {len(final_summary.objects)}")
        self.assertEqual(final_summary.objects[0].name, "Ball")
        self.assertEqual(len(final_summary.people), 3, f"Expected 3 people in FinalSummary, got {len(final_summary.people)}")

    def test_two_distinct_balls_preservation(self):
        """
        Verify that 2 physically distinct soccer balls are maintained as 2 distinct tracks.
        """
        tracker = SpatialIoUTracker()
        sampled_frames = [
            SampledFrame(frame_id="f1", timestamp=1.0, frame_index=30, path="f1.jpg", scene_id=1),
            SampledFrame(frame_id="f2", timestamp=2.0, frame_index=60, path="f2.jpg", scene_id=1),
        ]
        # Frame with 2 balls simultaneously visible at distinct locations
        frame_detections = {
            "f1": [
                YOLODetection(class_name="sports ball", confidence=0.90, bbox=[20.0, 20.0, 40.0, 40.0]),
                YOLODetection(class_name="sports ball", confidence=0.88, bbox=[250.0, 250.0, 270.0, 270.0]),
            ],
            "f2": [
                YOLODetection(class_name="sports ball", confidence=0.91, bbox=[22.0, 22.0, 42.0, 42.0]),
                YOLODetection(class_name="sports ball", confidence=0.89, bbox=[252.0, 252.0, 272.0, 272.0]),
            ],
        }
        tracks = tracker.track_entities(sampled_frames, frame_detections, [])
        ball_tracks = [t for t in tracks if t.canonical_name == "sports ball"]
        self.assertEqual(len(ball_tracks), 2, f"Expected 2 distinct ball tracks when 2 balls are present, got {len(ball_tracks)}")


if __name__ == "__main__":
    unittest.main()
