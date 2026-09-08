from pathlib import Path
from typing import List, Union, Dict
import torch

from config.settings import settings
from models.schemas import YOLODetection
from utils.logger import logger
from utils.caching import cache_manager

_GLOBAL_YOLO_MODEL = None


def get_shared_yolo_model(model_name: str = "yolov8m.pt"):
    """Load one YOLO model for the application lifetime."""
    global _GLOBAL_YOLO_MODEL
    if _GLOBAL_YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
            _GLOBAL_YOLO_MODEL = YOLO(model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                f"[YOLO] Loaded '{model_name}' on {device}. "
                "Using high-accuracy person/object detection."
            )
        except Exception as e:
            logger.exception(f"[YOLO] Model load failed: {e}")
            _GLOBAL_YOLO_MODEL = False
    return _GLOBAL_YOLO_MODEL if _GLOBAL_YOLO_MODEL is not False else None


class YOLOObjectDetector:
    """High-accuracy YOLO detector with video-versioned caching."""

    def __init__(self, model_name: str = "yolov8m.pt", confidence_threshold: float = None):
        self.model_name = model_name
        # 0.35 is a better recall/precision starting point for people in video.
        # The caller can still override it from the UI/settings.
        self.conf_thresh = (
            confidence_threshold
            if confidence_threshold is not None
            else getattr(settings, "YOLO_CONFIDENCE", 0.35)
        )

    @property
    def model(self):
        return get_shared_yolo_model(self.model_name)

    def detect_objects(
        self,
        image_path: Union[str, Path],
        video_hash: str = "",
        confidence_threshold: float = None,
    ) -> List[YOLODetection]:
        results = self.detect_objects_batch(
            [image_path], video_hash=video_hash,
            confidence_threshold=confidence_threshold,
        )
        return results.get(str(Path(image_path).resolve()), [])

    def detect_objects_batch(
        self,
        image_paths: List[Union[str, Path]],
        video_hash: str = "",
        confidence_threshold: float = None,
    ) -> Dict[str, List[YOLODetection]]:
        """Run deterministic batched YOLO detection over sampled video frames."""
        conf_val = (
            confidence_threshold
            if confidence_threshold is not None
            else self.conf_thresh
        )
        batch_results: Dict[str, List[YOLODetection]] = {}
        uncached_paths: List[Path] = []

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
            if cached is not None:
                batch_results[str(path)] = [YOLODetection(**d) for d in cached]
            else:
                uncached_paths.append(path)

        if not uncached_paths:
            return batch_results

        model = self.model
        if model is None:
            return batch_results

        try:
            # Keep image order exactly equal to the input frame order.
            results = model.predict(
                source=[str(p) for p in uncached_paths],
                conf=conf_val,
                iou=0.50,
                imgsz=960,
                max_det=300,
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=False,
            )

            for path, result in zip(uncached_paths, results):
                detections: List[YOLODetection] = []
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = result.names[cls_id]
                        confidence = float(box.conf[0].item())
                        xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]

                        detections.append(
                            YOLODetection(
                                class_name=cls_name,
                                confidence=round(confidence, 3),
                                bbox=xyxy,
                            )
                        )

                # Stable ordering makes downstream association deterministic.
                detections.sort(
                    key=lambda d: (
                        d.class_name,
                        d.bbox[0],
                        d.bbox[1],
                        -d.confidence,
                    )
                )
                path_str = str(path)
                batch_results[path_str] = detections

                cache_key = cache_manager.build_versioned_key(
                    prefix=f"yolo_det_{path.name}",
                    video_hash=video_hash,
                    model_name=f"{self.model_name}_conf{conf_val:.2f}",
                )
                cache_manager.set(cache_key, [d.model_dump() for d in detections])
                logger.info(
                    f"YOLO: {path.name}: {len(detections)} detections "
                    f"(conf={conf_val:.2f}, imgsz=960)"
                )

        except Exception as e:
            logger.exception(f"YOLO batch detection error: {e}")
            for path in uncached_paths:
                batch_results.setdefault(str(path), [])

        return batch_results


object_detector = YOLOObjectDetector()
