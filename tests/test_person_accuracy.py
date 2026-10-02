import unittest
from models.schemas import SampledFrame, YOLODetection, FrameObservation, VideoMetadata, Scene
from vision.tracker import SpatialIoUTracker
from vision.person_registry import canonical_person_registry, PersonEntity
from intelligence.temporal_reasoner import TemporalReasoner


class TestPersonAccuracy(unittest.TestCase):

    def test_1_one_stationary_person(self):
        """Test 1: One stationary person with bbox jitter yields exactly 1 canonical person."""
        tracker = SpatialIoUTracker()
        sampled_frames = [
            SampledFrame(frame_id="f1", timestamp=1.0, frame_index=30, path="f1.jpg", scene_id=1),
            SampledFrame(frame_id="f2", timestamp=3.0, frame_index=90, path="f2.jpg", scene_id=1),
            SampledFrame(frame_id="f3", timestamp=5.0, frame_index=150, path="f3.jpg", scene_id=1),
        ]
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

    def test_2_one_moving_person(self):
        """Test 2: One moving person walking across frame yields exactly 1 canonical person."""
        tracker = SpatialIoUTracker()
        sampled_frames = [
            SampledFrame(frame_id="f1", timestamp=1.0, frame_index=30, path="f1.jpg", scene_id=1),
            SampledFrame(frame_id="f2", timestamp=3.0, frame_index=90, path="f2.jpg", scene_id=1),
            SampledFrame(frame_id="f3", timestamp=5.0, frame_index=150, path="f3.jpg", scene_id=1),
        ]
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

    def test_3_five_stationary_people(self):
        """Test 3: Five stationary people across frames yield exactly 5 canonical people."""
        tracker = SpatialIoUTracker()
        sampled_frames = [
            SampledFrame(frame_id="f1", timestamp=1.0, frame_index=30, path="f1.jpg", scene_id=1),
            SampledFrame(frame_id="f2", timestamp=2.0, frame_index=60, path="f2.jpg", scene_id=1),
            SampledFrame(frame_id="f3", timestamp=3.0, frame_index=90, path="f3.jpg", scene_id=1),
        ]
        frame_detections = {}
        for fid in ["f1", "f2", "f3"]:
            dets = []
            for p in range(5):
                x1 = 40.0 + p * 100.0
                dets.append(YOLODetection(class_name="person", confidence=0.90, bbox=[x1, 100.0, x1 + 50.0, 250.0]))
            frame_detections[fid] = dets

        tracks = tracker.track_entities(sampled_frames, frame_detections, [])
        metadata = VideoMetadata(
            filename="five_static.mp4", filepath="five_static.mp4", file_size_mb=1.0,
            duration_sec=4.0, fps=30.0, width=640, height=480, frame_count=120, codec="h264", video_hash="fs555"
        )
        persons = canonical_person_registry.reconcile_person_tracks(tracks, [], metadata)

        self.assertEqual(len(persons), 5, f"Expected 5 canonical people for 5 stationary people, got {len(persons)}")

    def test_4_five_moving_people(self):
        """Test 4: Five moving people walking across frames yield approximately 5 canonical people with stable IDs."""
        tracker = SpatialIoUTracker()
        sampled_frames = [
            SampledFrame(frame_id=f"f{i}", timestamp=float(i), frame_index=i*30, path=f"f{i}.jpg", scene_id=1)
            for i in range(1, 6)
        ]
        frame_detections = {}
        for idx, sf in enumerate(sampled_frames):
            dets = []
            for p in range(5):
                x1 = 30.0 + p * 110.0 + idx * 15.0  # Movement shift
                dets.append(YOLODetection(class_name="person", confidence=0.88, bbox=[x1, 100.0, x1 + 50.0, 250.0]))
            frame_detections[sf.frame_id] = dets

        tracks = tracker.track_entities(sampled_frames, frame_detections, [])
        metadata = VideoMetadata(
            filename="five_moving.mp4", filepath="five_moving.mp4", file_size_mb=2.0,
            duration_sec=6.0, fps=30.0, width=640, height=480, frame_count=180, codec="h264", video_hash="fm555"
        )
        persons = canonical_person_registry.reconcile_person_tracks(tracks, [], metadata)

        self.assertEqual(len(persons), 5, f"Expected 5 canonical people for 5 moving people, got {len(persons)}")

    def test_5_five_people_two_overlap(self):
        """Test 5: Five people where two temporarily overlap yields exactly 5 canonical people (not 6+)."""
        tracker = SpatialIoUTracker()
        sampled_frames = [
            SampledFrame(frame_id="f1", timestamp=1.0, frame_index=30, path="f1.jpg", scene_id=1),
            SampledFrame(frame_id="f2", timestamp=2.0, frame_index=60, path="f2.jpg", scene_id=1),
            SampledFrame(frame_id="f3", timestamp=3.0, frame_index=90, path="f3.jpg", scene_id=1),
        ]
        # Frame 2: Person 1 and Person 2 cross near x=150
        frame_detections = {
            "f1": [
                YOLODetection(class_name="person", confidence=0.90, bbox=[50.0, 100.0, 100.0, 250.0]),
                YOLODetection(class_name="person", confidence=0.91, bbox=[250.0, 100.0, 300.0, 250.0]),
                YOLODetection(class_name="person", confidence=0.89, bbox=[350.0, 100.0, 400.0, 250.0]),
                YOLODetection(class_name="person", confidence=0.92, bbox=[450.0, 100.0, 500.0, 250.0]),
                YOLODetection(class_name="person", confidence=0.90, bbox=[550.0, 100.0, 600.0, 250.0]),
            ],
            "f2": [
                # Person 1 & 2 overlap near x=150
                YOLODetection(class_name="person", confidence=0.92, bbox=[140.0, 100.0, 190.0, 250.0]),
                YOLODetection(class_name="person", confidence=0.89, bbox=[350.0, 100.0, 400.0, 250.0]),
                YOLODetection(class_name="person", confidence=0.91, bbox=[450.0, 100.0, 500.0, 250.0]),
                YOLODetection(class_name="person", confidence=0.90, bbox=[550.0, 100.0, 600.0, 250.0]),
            ],
            "f3": [
                YOLODetection(class_name="person", confidence=0.91, bbox=[250.0, 100.0, 300.0, 250.0]),
                YOLODetection(class_name="person", confidence=0.90, bbox=[50.0, 100.0, 100.0, 250.0]),
                YOLODetection(class_name="person", confidence=0.89, bbox=[350.0, 100.0, 400.0, 250.0]),
                YOLODetection(class_name="person", confidence=0.92, bbox=[450.0, 100.0, 500.0, 250.0]),
                YOLODetection(class_name="person", confidence=0.90, bbox=[550.0, 100.0, 600.0, 250.0]),
            ]
        }
        tracks = tracker.track_entities(sampled_frames, frame_detections, [])
        metadata = VideoMetadata(
            filename="overlap.mp4", filepath="overlap.mp4", file_size_mb=1.0,
            duration_sec=4.0, fps=30.0, width=640, height=480, frame_count=120, codec="h264", video_hash="ov555"
        )
        persons = canonical_person_registry.reconcile_person_tracks(tracks, [], metadata)

        self.assertEqual(len(persons), 5, f"Expected 5 canonical people when two overlap, got {len(persons)}")

    def test_6_person_occlusion_reassociation(self):
        """Test 6: Person disappears temporarily behind another person -> retains original identity after reappearing."""
        tracker = SpatialIoUTracker()
        sampled_frames = [
            SampledFrame(frame_id="f1", timestamp=1.0, frame_index=30, path="f1.jpg", scene_id=1),
            SampledFrame(frame_id="f2", timestamp=2.0, frame_index=60, path="f2.jpg", scene_id=1),
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

        self.assertEqual(len(persons), 1, f"Expected 1 PersonEntity after occlusion reassociation, got {len(persons)}")
        self.assertEqual(persons[0].person_id, "Person #1")

    def test_7_fast_moving_person(self):
        """Test 7: Fast-moving person retains same identity rather than spawning duplicate tracks."""
        tracker = SpatialIoUTracker()
        sampled_frames = [
            SampledFrame(frame_id="f1", timestamp=1.0, frame_index=30, path="f1.jpg", scene_id=1),
            SampledFrame(frame_id="f2", timestamp=2.0, frame_index=60, path="f2.jpg", scene_id=1),
            SampledFrame(frame_id="f3", timestamp=3.0, frame_index=90, path="f3.jpg", scene_id=1),
        ]
        # Fast movement: x=50 -> x=200 -> x=380 (150px per second)
        frame_detections = {
            "f1": [YOLODetection(class_name="person", confidence=0.91, bbox=[50.0, 100.0, 100.0, 250.0])],
            "f2": [YOLODetection(class_name="person", confidence=0.90, bbox=[200.0, 100.0, 250.0, 250.0])],
            "f3": [YOLODetection(class_name="person", confidence=0.89, bbox=[380.0, 100.0, 430.0, 250.0])],
        }
        tracks = tracker.track_entities(sampled_frames, frame_detections, [])
        metadata = VideoMetadata(
            filename="fast_person.mp4", filepath="fast_person.mp4", file_size_mb=1.0,
            duration_sec=4.0, fps=30.0, width=640, height=480, frame_count=120, codec="h264", video_hash="fp123"
        )
        persons = canonical_person_registry.reconcile_person_tracks(tracks, [], metadata)

        self.assertEqual(len(persons), 1, f"Expected 1 canonical person for fast moving entity, got {len(persons)}")

    def test_8_false_one_frame_detection(self):
        """Test 8: Spurious one-frame false detection is rejected and not counted as a canonical person."""
        tracker = SpatialIoUTracker(min_confirmed_hits=2)
        sampled_frames = [
            SampledFrame(frame_id="f1", timestamp=1.0, frame_index=30, path="f1.jpg", scene_id=1),
            SampledFrame(frame_id="f2", timestamp=2.0, frame_index=60, path="f2.jpg", scene_id=1),
            SampledFrame(frame_id="f3", timestamp=3.0, frame_index=90, path="f3.jpg", scene_id=1),
        ]
        # Real Person 1 present in all 3 frames; False Person 2 only appears in 1 frame (f2) with low confidence
        frame_detections = {
            "f1": [YOLODetection(class_name="person", confidence=0.92, bbox=[100.0, 100.0, 150.0, 250.0])],
            "f2": [
                YOLODetection(class_name="person", confidence=0.91, bbox=[102.0, 100.0, 152.0, 250.0]),
                YOLODetection(class_name="person", confidence=0.52, bbox=[450.0, 300.0, 480.0, 350.0]),  # False 1-frame detection
            ],
            "f3": [YOLODetection(class_name="person", confidence=0.90, bbox=[101.0, 100.0, 151.0, 250.0])],
        }
        tracks = tracker.track_entities(sampled_frames, frame_detections, [])
        metadata = VideoMetadata(
            filename="spurious.mp4", filepath="spurious.mp4", file_size_mb=1.0,
            duration_sec=4.0, fps=30.0, width=640, height=480, frame_count=120, codec="h264", video_hash="sp888"
        )
        persons = canonical_person_registry.reconcile_person_tracks(tracks, [], metadata)

        self.assertEqual(len(persons), 1, f"Expected false 1-frame detection to be rejected (1 person total), got {len(persons)}")

    def test_9_shortnote_person_description(self):
        """Test 9: Verify that person entity generates a succinct short note description."""
        person = PersonEntity(
            person_id="Person #1",
            first_seen=0.0,
            last_seen=5.0,
            motion_state="MOVING",
            movement_distance=120.0,
            activities=["walking across room"],
            interactions=["holding cup"],
        )
        shortnote = person.get_shortnote_description()
        self.assertEqual(shortnote, "Walking across room")
        self.assertNotIn("Displacement:", shortnote)
        self.assertNotIn("Stance:", shortnote)
    def test_10_get_unique_person_count_authority(self):
        """Test 10: Verify get_unique_person_count returns authoritative count ignoring VLM text and raw tracks."""
        from vision.person_registry import get_unique_person_count
        tracker = SpatialIoUTracker()
        sampled_frames = [
            SampledFrame(frame_id="f1", timestamp=1.0, frame_index=30, path="f1.jpg", scene_id=1),
            SampledFrame(frame_id="f2", timestamp=2.0, frame_index=60, path="f2.jpg", scene_id=1),
        ]
        frame_detections = {
            "f1": [
                YOLODetection(class_name="person", confidence=0.92, bbox=[100.0, 100.0, 150.0, 250.0]),
                YOLODetection(class_name="person", confidence=0.90, bbox=[300.0, 100.0, 350.0, 250.0]),
            ],
            "f2": [
                YOLODetection(class_name="person", confidence=0.91, bbox=[102.0, 100.0, 152.0, 250.0]),
                YOLODetection(class_name="person", confidence=0.89, bbox=[302.0, 100.0, 352.0, 250.0]),
            ],
        }
        tracks = tracker.track_entities(sampled_frames, frame_detections, [])
        count = get_unique_person_count(tracks)
        self.assertEqual(count, 2, f"Expected get_unique_person_count() to return 2, got {count}")

    def test_11_six_people_manjha_isolation(self):
        """Test 11: 6 people where only Person #1 holds manjha. Persons #2-#6 must NOT inherit manjha."""
        person1 = PersonEntity(
            person_id="Person #1", first_seen=1.0, last_seen=5.0,
            objects_held=["manjha"], person_description="Holding manjha."
        )
        person2 = PersonEntity(person_id="Person #2", first_seen=1.0, last_seen=5.0, motion_state="STATIONARY")
        person3 = PersonEntity(person_id="Person #3", first_seen=1.0, last_seen=5.0, motion_state="STATIONARY")
        person4 = PersonEntity(person_id="Person #4", first_seen=1.0, last_seen=5.0, motion_state="MOVING")
        person5 = PersonEntity(person_id="Person #5", first_seen=1.0, last_seen=5.0, motion_state="STATIONARY")
        person6 = PersonEntity(person_id="Person #6", first_seen=1.0, last_seen=5.0, motion_state="STATIONARY")

        people = [person1, person2, person3, person4, person5, person6]

        self.assertIn("manjha", person1.get_shortnote_description().lower())
        for p in people[1:]:
            desc = p.get_shortnote_description().lower()
            self.assertNotIn("manjha", desc, f"{p.person_id} falsely inherited manjha: {desc}")
            self.assertEqual(len(p.objects_held), 0, f"{p.person_id} falsely has objects_held")

    def test_12_different_held_objects_isolation(self):
        """Test 12: Verify isolated object ownership per person without cross-contamination."""
        p1 = PersonEntity(person_id="Person #1", first_seen=1.0, last_seen=5.0, objects_held=["manjha"])
        p2 = PersonEntity(person_id="Person #2", first_seen=1.0, last_seen=5.0, objects_held=["mobile phone"])
        p3 = PersonEntity(person_id="Person #3", first_seen=1.0, last_seen=5.0, objects_held=["water bottle"])
        p4 = PersonEntity(person_id="Person #4", first_seen=1.0, last_seen=5.0, motion_state="STATIONARY")

        self.assertEqual(p1.get_shortnote_description(), "Holding manjha.")
        self.assertEqual(p2.get_shortnote_description(), "Holding mobile phone.")
        self.assertEqual(p3.get_shortnote_description(), "Holding water bottle.")
        self.assertIn("No clearly visible held object", p4.get_shortnote_description())


if __name__ == "__main__":
    unittest.main()

