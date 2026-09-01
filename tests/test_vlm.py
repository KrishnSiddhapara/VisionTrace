import unittest
import tempfile
from pathlib import Path
from PIL import Image
from models.schemas import SampledFrame
from vision.vlm import MockVLMProvider
from vision.frame_analyzer import FrameAnalyzer

class TestVLM(unittest.TestCase):

    def test_mock_vlm_frame_analyzer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "test_frame.jpg"
            img = Image.new("RGB", (200, 200), color="blue")
            img.save(img_path)

            sampled = SampledFrame(
                frame_id="frame_0001",
                timestamp=1.5,
                frame_index=45,
                path=str(img_path),
                scene_id=1,
                sampling_reason="uniform"
            )

            analyzer = FrameAnalyzer(vlm_provider=MockVLMProvider())
            obs = analyzer.analyze_frame(sampled)

            self.assertEqual(obs.frame_id, "frame_0001")
            self.assertEqual(obs.timestamp, 1.5)
            self.assertNotEqual(obs.environment, "")
            self.assertGreater(len(obs.people), 0)
            self.assertGreater(len(obs.objects), 0)

if __name__ == "__main__":
    unittest.main()
