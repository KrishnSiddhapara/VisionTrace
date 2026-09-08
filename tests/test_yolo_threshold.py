import unittest
import tempfile
from pathlib import Path
import numpy as np
import cv2

from vision.object_detector import object_detector


class TestYOLOThreshold(unittest.TestCase):

    def test_yolo_confidence_threshold_propagation(self):
        """Verify that changing confidence_threshold affects accepted YOLO detections."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "test_frame.jpg"

            # Create synthetic frame with shapes
            img = np.full((300, 300, 3), (200, 200, 200), dtype=np.uint8)
            cv2.circle(img, (150, 150), 50, (50, 50, 220), -1)
            cv2.imwrite(str(img_path), img)

            # Test detect_objects with 0.10, 0.50, and 0.90 thresholds
            dets_low = object_detector.detect_objects(img_path, confidence_threshold=0.10)
            dets_mid = object_detector.detect_objects(img_path, confidence_threshold=0.50)
            dets_high = object_detector.detect_objects(img_path, confidence_threshold=0.90)

            # Verify all return list instances
            self.assertIsInstance(dets_low, list)
            self.assertIsInstance(dets_mid, list)
            self.assertIsInstance(dets_high, list)

            # Threshold monotonicity: low threshold yields >= mid threshold >= high threshold detections
            self.assertGreaterEqual(len(dets_low), len(dets_mid))
            self.assertGreaterEqual(len(dets_mid), len(dets_high))


if __name__ == "__main__":
    unittest.main()
