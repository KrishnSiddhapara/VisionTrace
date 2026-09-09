import unittest
import numpy as np
from models.schemas import SampledFrame, YOLODetection
from vision.tracker import SpatialIoUTracker
from video.movement_detector import OpenCVMovementDetector

class TestTrackingAccuracy(unittest.TestCase):

    def setUp(self):
        self.tracker = SpatialIoUTracker(min_confirmed_hits=2, max_lost_frames=10, max_lost_seconds=3.0)
        self.movement_detector = OpenCVMovementDetector()

    def test_7_people_unique_identity(self):
        """Scenario 1: 7 distinct people across 10 frames -> must yield exactly 7 unique people."""
        sampled_frames = []
        frame_dets = {}

        for i in range(10):
            fid = f"frame_{i:04d}"
            ts = round(i * 0.5, 2)
            sf = SampledFrame(
                frame_id=fid, timestamp=ts, frame_index=i*5, path=f"dummy_{fid}.jpg", scene_id=1
            )
            sampled_frames.append(sf)

            # 7 people moving slightly to the right in each frame
            dets = []
            for p in range(7):
                x1 = 50.0 + p * 100.0 + i * 2.0
                y1 = 100.0
                x2 = x1 + 40.0
                y2 = 190.0
                dets.append(
                    YOLODetection(
                        class_name="person",
                        confidence=0.90,
                        bbox=[x1, y1, x2, y2],
                        center_x=x1 + 20.0,
                        center_y=145.0,
                        width=40.0,
                        height=90.0,
                        area=3600.0,
                    )
                )
            frame_dets[fid] = dets

        tracks = self.tracker.track_entities(sampled_frames, frame_dets)
        people_tracks = [t for t in tracks if t.canonical_name == "person"]

        self.assertEqual(len(people_tracks), 7, f"Expected 7 unique people, got {len(people_tracks)}")

    def test_3_people_1_soccer_ball(self):
        """Scenario 2: 3 people + 1 soccer ball moving across frames -> 3 unique people, 1 Ball #1."""
        sampled_frames = []
        frame_dets = {}

        for i in range(8):
            fid = f"frame_{i:04d}"
            ts = round(i * 0.5, 2)
            sf = SampledFrame(
                frame_id=fid, timestamp=ts, frame_index=i*5, path=f"dummy_{fid}.jpg", scene_id=1
            )
            sampled_frames.append(sf)

            dets = []
            # 3 people
            for p in range(3):
                x1 = 40.0 + p * 150.0
                y1 = 120.0
                dets.append(
                    YOLODetection(
                        class_name="person", confidence=0.92, bbox=[x1, y1, x1 + 50.0, y1 + 120.0]
                    )
                )
            # 1 ball moving rapidly
            bx1 = 100.0 + i * 45.0
            by1 = 200.0
            dets.append(
                YOLODetection(
                    class_name="soccer ball", confidence=0.88, bbox=[bx1, by1, bx1 + 25.0, by1 + 25.0]
                )
            )
            frame_dets[fid] = dets

        tracks = self.tracker.track_entities(sampled_frames, frame_dets)
        people_tracks = [t for t in tracks if t.canonical_name == "person"]
        ball_tracks = [t for t in tracks if t.canonical_name == "sports ball"]

        self.assertEqual(len(people_tracks), 3, f"Expected 3 unique people, got {len(people_tracks)}")
        self.assertEqual(len(ball_tracks), 1, f"Expected 1 soccer ball, got {len(ball_tracks)}")
        self.assertIn("Ball", ball_tracks[0].track_id)

    def test_person_enters_scene(self):
        """Scenario 3: Start 3 people, middle 4th person enters -> unique_people == 4."""
        sampled_frames = []
        frame_dets = {}

        for i in range(10):
            fid = f"frame_{i:04d}"
            ts = round(i * 0.5, 2)
            sf = SampledFrame(frame_id=fid, timestamp=ts, frame_index=i*5, path=f"dummy_{fid}.jpg", scene_id=1)
            sampled_frames.append(sf)

            dets = []
            # Initial 3 people always visible
            for p in range(3):
                x1 = 50.0 + p * 120.0
                dets.append(YOLODetection(class_name="person", confidence=0.91, bbox=[x1, 80.0, x1 + 45.0, 180.0]))

            # 4th person enters at frame 4 (index >= 4)
            if i >= 4:
                x1 = 450.0 + (i - 4) * 10.0
                dets.append(YOLODetection(class_name="person", confidence=0.89, bbox=[x1, 80.0, x1 + 45.0, 180.0]))

            frame_dets[fid] = dets

        tracks = self.tracker.track_entities(sampled_frames, frame_dets)
        people_tracks = [t for t in tracks if t.canonical_name == "person"]

        self.assertEqual(len(people_tracks), 4, f"Expected 4 unique people, got {len(people_tracks)}")

    def test_person_leaves_scene(self):
        """Scenario 4: Start 7 people, 1 person leaves -> unique_people == 7, last_seen accurately tracked."""
        sampled_frames = []
        frame_dets = {}

        for i in range(10):
            fid = f"frame_{i:04d}"
            ts = round(i * 0.5, 2)
            sf = SampledFrame(frame_id=fid, timestamp=ts, frame_index=i*5, path=f"dummy_{fid}.jpg", scene_id=1)
            sampled_frames.append(sf)

            # Frame 0-4: 7 people. Frame 5-9: Person #7 leaves.
            active_count = 7 if i < 5 else 6
            dets = []
            for p in range(active_count):
                x1 = 40.0 + p * 80.0
                dets.append(YOLODetection(class_name="person", confidence=0.90, bbox=[x1, 90.0, x1 + 40.0, 190.0]))

            frame_dets[fid] = dets

        tracks = self.tracker.track_entities(sampled_frames, frame_dets)
        people_tracks = [t for t in tracks if t.canonical_name == "person"]

        self.assertEqual(len(people_tracks), 7, f"Expected 7 unique people total, got {len(people_tracks)}")
        left_person = [t for t in people_tracks if t.last_seen <= 2.0]
        self.assertEqual(len(left_person), 1, "Expected 1 person track to have early last_seen timestamp")

    def test_people_crossing_no_identity_swap(self):
        """Scenario 5: Two people cross each other horizontally -> 1-to-1 Hungarian matching preserves identities."""
        sampled_frames = []
        frame_dets = {}

        # Frame 0: P1 at x=50, P2 at x=250
        # Frame 1: P1 at x=100, P2 at x=200
        # Frame 2: P1 at x=145, P2 at x=155 (Crossing)
        # Frame 3: P1 at x=200, P2 at x=100
        # Frame 4: P1 at x=250, P2 at x=50
        p1_x_coords = [50.0, 100.0, 145.0, 200.0, 250.0]
        p2_x_coords = [250.0, 200.0, 155.0, 100.0, 50.0]

        for i in range(5):
            fid = f"frame_{i:04d}"
            ts = round(i * 0.5, 2)
            sf = SampledFrame(frame_id=fid, timestamp=ts, frame_index=i*5, path=f"dummy_{fid}.jpg", scene_id=1)
            sampled_frames.append(sf)

            d1 = YOLODetection(class_name="person", confidence=0.93, bbox=[p1_x_coords[i], 100.0, p1_x_coords[i] + 40.0, 200.0])
            d2 = YOLODetection(class_name="person", confidence=0.92, bbox=[p2_x_coords[i], 100.0, p2_x_coords[i] + 40.0, 200.0])
            frame_dets[fid] = [d1, d2]

        tracks = self.tracker.track_entities(sampled_frames, frame_dets)
        people_tracks = [t for t in tracks if t.canonical_name == "person"]

        self.assertEqual(len(people_tracks), 2, f"Expected 2 unique people, got {len(people_tracks)}")

        # Check trajectory continuity for Person #1 (moving left to right)
        p1_track = [t for t in people_tracks if t.positions[0]["bbox"][0] < 100.0][0]
        self.assertGreater(p1_track.positions[-1]["bbox"][0], 200.0, "Person #1 should remain Person #1 after crossing")

    def test_occlusion_recovery(self):
        """Scenario 6: Person disappears for 2 frames during occlusion, then reappears -> retains original track ID."""
        sampled_frames = []
        frame_dets = {}

        for i in range(8):
            fid = f"frame_{i:04d}"
            ts = round(i * 0.5, 2)
            sf = SampledFrame(frame_id=fid, timestamp=ts, frame_index=i*5, path=f"dummy_{fid}.jpg", scene_id=1)
            sampled_frames.append(sf)

            dets = []
            # Person #1 always visible
            dets.append(YOLODetection(class_name="person", confidence=0.95, bbox=[50.0, 100.0, 90.0, 200.0]))

            # Person #2 invisible during frames 3 and 4 (occluded), reappears frame 5
            if i not in (3, 4):
                x2 = 200.0 + i * 5.0
                dets.append(YOLODetection(class_name="person", confidence=0.90, bbox=[x2, 100.0, x2 + 40.0, 200.0]))

            frame_dets[fid] = dets

        tracks = self.tracker.track_entities(sampled_frames, frame_dets)
        people_tracks = [t for t in tracks if t.canonical_name == "person"]

        self.assertEqual(len(people_tracks), 2, f"Occlusion recovery failed! Expected 2 people, got {len(people_tracks)}")

    def test_camera_motion_compensation(self):
        """Scenario 8: Camera translation shift subtracts camera motion from bounding box displacement."""
        bbox = [100.0, 100.0, 150.0, 200.0]
        shift_x, shift_y = 20.0, 5.0

        comp_bbox = self.movement_detector.compensate_camera_motion(bbox, shift_x, shift_y)
        self.assertEqual(comp_bbox, [80.0, 95.0, 130.0, 195.0])

if __name__ == "__main__":
    unittest.main()
