from pathlib import Path
from typing import List, Union, Dict, Any, Optional
import cv2
import torch

from config.settings import settings
from models.schemas import YOLODetection
from utils.logger import logger
from utils.caching import cache_manager

_GLOBAL_YOLO_MODEL = None

def get_shared_yolo_model(model_name: str = "yolov8n.pt"):
    """Load YOLO model instance ONLY ONCE across the entire application runtime."""
    global _GLOBAL_YOLO_MODEL
    if _GLOBAL_YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
            _GLOBAL_YOLO_MODEL = YOLO(model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[YOLO Singleton] Loaded YOLO model '{model_name}' on device '{device}'.")
        except Exception as e:
            logger.warning(f"[YOLO Singleton] Could not load YOLO model ({e}). Fallback mode active.")
            _GLOBAL_YOLO_MODEL = False
    return _GLOBAL_YOLO_MODEL if _GLOBAL_YOLO_MODEL is not False else None


class YOLOObjectDetector:
    """Centralized YOLO Object Detector with batch inference, caching, and threshold control."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = None):
        self.model_name = model_name
        self.conf_thresh = confidence_threshold or settings.YOLO_CONFIDENCE

    @property
    def model(self):
        return get_shared_yolo_model(self.model_name)

    def detect_objects(
        self,
        image_path: Union[str, Path],
        video_hash: str = "",
        confidence_threshold: float = None
    ) -> List[YOLODetection]:
        results_map = self.detect_objects_batch([image_path], video_hash=video_hash, confidence_threshold=confidence_threshold)
        return results_map.get(str(Path(image_path).resolve()), [])

    def detect_objects_batch(
        self,
        image_paths: List[Union[str, Path]],
        video_hash: str = "",
        confidence_threshold: float = None
    ) -> Dict[str, List[YOLODetection]]:
        """
        Batch YOLO inference for multiple frames.
        Reuses versioned cache and executes batched model.predict() in a single call.
        """
        conf_val = confidence_threshold if confidence_threshold is not None else self.conf_thresh
        batch_results: Dict[str, List[YOLODetection]] = {}
        uncached_paths: List[Path] = []

        # 1. Check versioned cache for each frame
        for ip in image_paths:
            path = Path(ip).resolve()
            if not path.exists():
                batch_results[str(path)] = []
                continue

            cache_key = cache_manager.build_versioned_key(
                prefix=f"yolo_det_{path.name}",
                video_hash=video_hash,
                model_name=f"{self.model_name}_conf{conf_val:.2f}",
            )
            cached = cache_manager.get(cache_key)
            if cached:
                batch_results[str(path)] = [YOLODetection(**d) for d in cached]
            else:
                uncached_paths.append(path)

        if not uncached_paths:
            return batch_results

        # 2. Run batched YOLO prediction on uncached frames
        model = self.model
        if model:
            try:
                sources = [str(p) for p in uncached_paths]
                # Run batch prediction
                results = model.predict(source=sources, conf=conf_val, verbose=False)
                
                for path, r in zip(uncached_paths, results):
                    path_str = str(path)
                    detections: List[YOLODetection] = []
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
                    batch_results[path_str] = detections

                    # Cache detection result
                    cache_key = cache_manager.build_versioned_key(
                        prefix=f"yolo_det_{path.name}",
                        video_hash=video_hash,
                        model_name=f"{self.model_name}_conf{conf_val:.2f}",
                    )
                    cache_manager.set(cache_key, [d.model_dump() for d in detections])
                    logger.info(f"YOLO inference on {path.name} (conf={conf_val:.2f}): {len(detections)} accepted detections")

            except Exception as e:
                logger.error(f"YOLO batch detection error: {e}")
                for p in uncached_paths:
                    if str(p) not in batch_results:
                        batch_results[str(p)] = []

        # 3. Fallback mock detection ONLY when explicit VLM_MOCK_MODE is True
        for p in uncached_paths:
            path_str = str(p)
            if path_str not in batch_results or not batch_results[path_str]:
                if settings.VLM_MOCK_MODE:
                    batch_results[path_str] = [
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

        return batch_results


object_detector = YOLOObjectDetector()
