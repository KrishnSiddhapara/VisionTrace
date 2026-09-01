from pathlib import Path
from typing import List, Union, Dict, Any
import cv2

from config.settings import settings
from models.schemas import YOLODetection
from utils.logger import logger
from utils.caching import cache_manager

class YOLOObjectDetector:
    """YOLO Object Detector using Ultralytics YOLO with fallback."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = None):
        self.model_name = model_name
        self.conf_thresh = confidence_threshold or settings.YOLO_CONFIDENCE
        self.model = None

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            logger.info(f"Loaded YOLO model {self.model_name} with threshold {self.conf_thresh}")
        except Exception as e:
            logger.warning(f"Could not initialize YOLO model ({e}). Fallback detector will be active.")

    def detect_objects(self, image_path: Union[str, Path], video_hash: str = "") -> List[YOLODetection]:
        path = Path(image_path)
        if not path.exists():
            return []

        cache_key = cache_manager.build_versioned_key(
            prefix=f"yolo_det_{path.name}",
            video_hash=video_hash,
            model_name=f"{self.model_name}_{self.conf_thresh}",
        )

        cached = cache_manager.get(cache_key)
        if cached:
            return [YOLODetection(**d) for d in cached]

        detections: List[YOLODetection] = []

        if self.model:
            try:
                results = self.model.predict(source=str(path), conf=self.conf_thresh, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = r.names[cls_id]
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].tolist()

                        detections.append(
                            YOLODetection(
                                class_name=cls_name,
                                confidence=round(conf, 3),
                                bbox=[round(v, 1) for v in xyxy],
                            )
                        )
            except Exception as e:
                logger.error(f"YOLO detection error on {path.name}: {e}")

        if not detections and settings.VLM_MOCK_MODE:
            # Fallback mock detection ONLY when explicit VLM_MOCK_MODE is True
            detections = [
                YOLODetection(
                    class_name="person",
                    confidence=0.94,
                    bbox=[100.0, 120.0, 300.0, 500.0],
                ),
                YOLODetection(
                    class_name="backpack",
                    confidence=0.87,
                    bbox=[250.0, 350.0, 380.0, 480.0],
                ),
            ]

        cache_manager.set(cache_key, [d.model_dump() for d in detections])
        return detections

object_detector = YOLOObjectDetector()
