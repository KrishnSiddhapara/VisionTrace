import unittest
from models.schemas import SampledFrame, YOLODetection, FrameObservation, VideoMetadata, Scene
from vision.tracker import SpatialIoUTracker
from vision.person_registry import canonical_person_registry, PersonEntity
from intelligence.temporal_reasoner import TemporalReasoner
from intelligence.event_detector import EventDetector


class TestPersonAccuracy(unittest.TestCase):

    def test_single_stationary_person_no_false_movement(self):
        """Test 1 stationary person with bounding box jitter does not trigger false movement."""
        tracker = SpatialIoUTracker()
        sampled_frames = [
            SampledFrame(frame_id="f1", timestamp=1.0, frame_index=30, path="f1.jpg", scene_id=1),
            SampledFrame(frame_id="f2", timestamp=3.0, frame_index=90, path="f2.jpg", scene_id=1),
            SampledFrame(frame_id="f3", timestamp=5.0, frame_index=150, path="f3.jpg", scene_id=1),
        ]
        # Bbox jitter: (100, 100) -> (103, 101) -> (101, 102) (displacement < 5px)
        frame_detections = {
            "f1": [YOLODetection(class_name="person", confidence=0.92, bbox=[100.0, 100.0, 160.0, 300.0])],
            "f2": [YOLODetection(class_name="person", confidence=0.90, bbox=[103.0, 101.0, 163.0, 301.0])],
            "f3": [YOLODetection(class_name="person", confidence=0.91, bbox=[101.0, 102.0, 161.0, 302.0])],
        }
        tracks = tracker.track_entities(sampled_frames, frame_detections, [])
        metadata = VideoMetadata(
            filename="static_person.mp4", filepath="static_person.mp4", file_size_mb=1.0,
            duration_sec=6.0, fps=30.0, width=640, height=480, frame_count=180, codec="h264", video_hash="sp123"
        )
        persons = canonical_person_registry.reconcile_person_tracks(tracks, [], metadata)

        self.assertEqual(len(persons), 1, f"Expected 1 person entity, got {len(persons)}")
        self.assertEqual(persons[0].motion_state, "STATIONARY", "Stationary person should have motion_state='STATIONARY'")

    def test_person_walking_movement_detected(self):
        """Test 1 person walking across screen is correctly classified as MOVING."""
        tracker = SpatialIoUTracker()
        sampled_frames = [
            SampledFrame(frame_id="f1", timestamp=1.0, frame_index=30, path="f1.jpg", scene_id=1),
            SampledFrame(frame_id="f2", timestamp=3.0, frame_index=90, path="f2.jpg", scene_id=1),
            SampledFrame(frame_id="f3", timestamp=5.0, frame_index=150, path="f3.jpg", scene_id=1),
        ]
        # Significant displacement: x=50 -> x=200 -> x=350 (displacement = 300px)
        frame_detections = {
            "f1": [YOLODetection(class_name="person", confidence=0.92, bbox=[50.0, 100.0, 110.0, 300.0])],
            "f2": [YOLODetection(class_name="person", confidence=0.90, bbox=[200.0, 100.0, 260.0, 300.0])],
            "f3": [YOLODetection(class_name="person", confidence=0.91, bbox=[350.0, 100.0, 410.0, 300.0])],
        }
        tracks = tracker.track_entities(sampled_frames, frame_detections, [])
        metadata = VideoMetadata(
            filename="walking_person.mp4", filepath="walking_person.mp4", file_size_mb=1.0,
            duration_sec=6.0, fps=30.0, width=640, height=480, frame_count=180, codec="h264", video_hash="wp123"
        )
        persons = canonical_person_registry.reconcile_person_tracks(tracks, [], metadata)

        self.assertEqual(len(persons), 1, f"Expected 1 person entity, got {len(persons)}")
        self.assertEqual(persons[0].motion_state, "MOVING", "Walking person should have motion_state='MOVING'")

    def test_occlusion_track_reassociation(self):
        """Test person track fragmented by temporary occlusion re-associates into 1 physical person identity."""
        tracker = SpatialIoUTracker()
        sampled_frames = [
            SampledFrame(frame_id="f1", timestamp=1.0, frame_index=30, path="f1.jpg", scene_id=1),
            SampledFrame(frame_id="f2", timestamp=2.0, frame_index=60, path="f2.jpg", scene_id=1),
            # Frame 3 (t=3s) missing due to occlusion
            SampledFrame(frame_id="f4", timestamp=4.0, frame_index=120, path="f4.jpg", scene_id=1),
            SampledFrame(frame_id="f5", timestamp=5.0, frame_index=150, path="f5.jpg", scene_id=1),
        ]

        frame_detections = {
            "f1": [YOLODetection(class_name="person", confidence=0.92, bbox=[100.0, 100.0, 160.0, 300.0])],
            "f2": [YOLODetection(class_name="person", confidence=0.90, bbox=[120.0, 100.0, 180.0, 300.0])],
            "f4": [YOLODetection(class_name="person", confidence=0.88, bbox=[160.0, 100.0, 220.0, 300.0])],
            "f5": [YOLODetection(class_name="person", confidence=0.91, bbox=[180.0, 100.0, 240.0, 300.0])],
        }

        tracks = tracker.track_entities(sampled_frames, frame_detections, [])
        metadata = VideoMetadata(
            filename="occluded_person.mp4", filepath="occluded_person.mp4", file_size_mb=1.0,
            duration_sec=6.0, fps=30.0, width=640, height=480, frame_count=180, codec="h264", video_hash="op123"
        )
        persons = canonical_person_registry.reconcile_person_tracks(tracks, [], metadata)

        self.assertEqual(len(persons), 1, f"Expected occlusion re-association to produce 1 PersonEntity, got {len(persons)}")
        self.assertEqual(persons[0].person_id, "Person #1")

    def test_three_people_soccer_game_count_consistency(self):
        """Test 3 people playing soccer game strictly reports 3 people in FinalSummary."""
        tracker = SpatialIoUTracker()
        reasoner = TemporalReasoner()

        sampled_frames = [
            SampledFrame(frame_id="f1", timestamp=1.0, frame_index=30, path="f1.jpg", scene_id=1),
            SampledFrame(frame_id="f2", timestamp=3.0, frame_index=90, path="f2.jpg", scene_id=1),
        ]
        frame_detections = {
            "f1": [
                YOLODetection(class_name="person", confidence=0.92, bbox=[50.0, 50.0, 100.0, 200.0]),
                YOLODetection(class_name="person", confidence=0.90, bbox=[150.0, 50.0, 200.0, 200.0]),
                YOLODetection(class_name="person", confidence=0.88, bbox=[250.0, 50.0, 300.0, 200.0]),
                YOLODetection(class_name="sports ball", confidence=0.85, bbox=[60.0, 180.0, 80.0, 200.0]),
            ],
            "f2": [
                YOLODetection(class_name="person", confidence=0.93, bbox=[55.0, 50.0, 105.0, 200.0]),
                YOLODetection(class_name="person", confidence=0.91, bbox=[155.0, 50.0, 205.0, 200.0]),
                YOLODetection(class_name="person", confidence=0.89, bbox=[255.0, 50.0, 305.0, 200.0]),
                YOLODetection(class_name="sports ball", confidence=0.84, bbox=[160.0, 180.0, 180.0, 200.0]),
            ],
        }
        tracks = tracker.track_entities(sampled_frames, frame_detections, [])
        metadata = VideoMetadata(
            filename="soccer.mp4", filepath="soccer.mp4", file_size_mb=2.0, duration_sec=4.0,
            fps=30.0, width=640, height=480, frame_count=120, codec="h264", video_hash="soc123"
        )
        scenes = [Scene(scene_id=1, start_time=0.0, end_time=4.0, duration=4.0, start_frame=0, end_frame=120)]
        timeline = reasoner.synthesize_timeline(scenes, [], tracks, [])
        final_summary = reasoner.generate_final_summary(metadata, scenes, timeline, tracks, [], sampled_frames)

        self.assertEqual(len(final_summary.people), 3, f"Expected 3 people in FinalSummary, got {len(final_summary.people)}")
        self.assertEqual(len(final_summary.objects), 1, f"Expected 1 ball object in FinalSummary, got {len(final_summary.objects)}")

    def test_four_children_vlm_recovery_when_yolo_detects_zero(self):
        """Test video with 4 children where YOLO detects 0 persons -> recovers 4 PersonEntity records from VLM observations."""
        from models.schemas import PersonObservation
        frame_observations = [
            FrameObservation(
                frame_id="f1", timestamp=1.0, scene_id=1, environment="Playground", is_analyzed=True,
                people=[
                    PersonObservation(temporary_id="Child #1", description="child in red shirt", activity="playing with blocks"),
                    PersonObservation(temporary_id="Child #2", description="child in blue shirt", activity="running on grass"),
                    PersonObservation(temporary_id="Child #3", description="child sitting down", activity="drawing"),
                    PersonObservation(temporary_id="Child #4", description="child wearing hat", activity="standing near slide"),
                ]
            ),
            FrameObservation(
                frame_id="f2", timestamp=3.0, scene_id=1, environment="Playground", is_analyzed=True,
                people=[
                    PersonObservation(temporary_id="Child #1", description="child in red shirt", activity="building tower"),
                    PersonObservation(temporary_id="Child #2", description="child in blue shirt", activity="playing tag"),
                    PersonObservation(temporary_id="Child #3", description="child sitting down", activity="coloring"),
                    PersonObservation(temporary_id="Child #4", description="child wearing hat", activity="climbing slide"),
                ]
            )
        ]
        metadata = VideoMetadata(
            filename="children_playing.mp4", filepath="children_playing.mp4", file_size_mb=3.0,
            duration_sec=6.0, fps=30.0, width=640, height=480, frame_count=180, codec="h264", video_hash="cp444"
        )
        persons = canonical_person_registry.reconcile_person_tracks([], frame_observations, metadata)
        self.assertEqual(len(persons), 4, f"Expected 4 children entities from VLM recovery, got {len(persons)}")
        self.assertEqual([p.person_id for p in persons], ["Person #1", "Person #2", "Person #3", "Person #4"])

    def test_four_children_vlm_recovery_when_yolo_detects_partial(self):
        """Test video with 4 children where YOLO detects 1 person track -> reconciles into 4 PersonEntity records."""
        from models.schemas import PersonObservation
        tracker = SpatialIoUTracker()
        sampled_frames = [
            SampledFrame(frame_id="f1", timestamp=1.0, frame_index=30, path="f1.jpg", scene_id=1),
            SampledFrame(frame_id="f2", timestamp=3.0, frame_index=90, path="f2.jpg", scene_id=1),
        ]
        # YOLO only detected 1 person out of 4 children
        frame_detections = {
            "f1": [YOLODetection(class_name="person", confidence=0.85, bbox=[50.0, 50.0, 100.0, 200.0])],
            "f2": [YOLODetection(class_name="person", confidence=0.86, bbox=[55.0, 50.0, 105.0, 200.0])],
        }
        tracks = tracker.track_entities(sampled_frames, frame_detections, [])

        frame_observations = [
            FrameObservation(
                frame_id="f1", timestamp=1.0, scene_id=1, environment="Room", is_analyzed=True,
                people=[
                    PersonObservation(temporary_id="Person #1", description="child 1", activity="playing"),
                    PersonObservation(temporary_id="Person #2", description="child 2", activity="playing"),
                    PersonObservation(temporary_id="Person #3", description="child 3", activity="playing"),
                    PersonObservation(temporary_id="Person #4", description="child 4", activity="playing"),
                ]
            )
        ]
        metadata = VideoMetadata(
            filename="children_partial.mp4", filepath="children_partial.mp4", file_size_mb=3.0,
            duration_sec=6.0, fps=30.0, width=640, height=480, frame_count=180, codec="h264", video_hash="cp123"
        )
        persons = canonical_person_registry.reconcile_person_tracks(tracks, frame_observations, metadata)
        self.assertEqual(len(persons), 4, f"Expected 4 children entities when YOLO detected 1 person, got {len(persons)}")

    def test_four_children_text_mention_extraction(self):
        """Test video where VLM text observation mentions 'Four children playing in the park' -> yields 4 entities."""
        frame_observations = [
            FrameObservation(
                frame_id="f1", timestamp=1.0, scene_id=1, environment="Park", is_analyzed=True,
                observations=["Four children are playing together in the park ground."],
                activities=["Four children running around."]
            )
        ]
        metadata = VideoMetadata(
            filename="children_text.mp4", filepath="children_text.mp4", file_size_mb=3.0,
            duration_sec=6.0, fps=30.0, width=640, height=480, frame_count=180, codec="h264", video_hash="cp999"
        )
        persons = canonical_person_registry.reconcile_person_tracks([], frame_observations, metadata)
        self.assertEqual(len(persons), 4, f"Expected 4 children entities from text observation extraction, got {len(persons)}")


if __name__ == "__main__":
    unittest.main()

