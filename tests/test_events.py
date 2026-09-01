import unittest
from models.schemas import Scene, FrameObservation, TrackedObject
from intelligence.event_detector import event_detector

class TestEvents(unittest.TestCase):

    def test_event_detection_engine(self):
        scenes = [
            Scene(scene_id=1, start_time=0.0, end_time=10.0, duration=10.0, start_frame=0, end_frame=300)
        ]
        frame_obs = [
            FrameObservation(
                frame_id="frame_0001",
                timestamp=2.5,
                scene_id=1,
                environment="Office",
                activities=["Person walking"],
            )
        ]
        tracks = [
            TrackedObject(
                track_id="person_01",
                object_type="person",
                first_seen=1.0,
                last_seen=9.0,
            )
        ]

        events = event_detector.detect_events(scenes, frame_obs, tracks)
        self.assertGreaterEqual(len(events), 3)
        event_types = {e.event_type for e in events}
        self.assertIn("SCENE", event_types)
        self.assertIn("PERSON", event_types)
        self.assertIn("MOVEMENT", event_types)

if __name__ == "__main__":
    unittest.main()
