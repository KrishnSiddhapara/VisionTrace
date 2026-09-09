from pathlib import Path
from typing import List, Union, Dict
import torch

from config.settings import settings
from models.schemas import YOLODetection
from utils.logger import logger
from utils.caching import cache_manager

_GLOBAL_YOLO_MODELS: Dict[str, Any] = {}

<<<<<<< HEAD
def get_shared_yolo_model(model_name: str = None):
    """Load YOLO model instance ONLY ONCE across the entire application runtime with fallback support."""
    global _GLOBAL_YOLO_MODELS
    target_name = model_name or settings.YOLO_MODEL
    
    if target_name in _GLOBAL_YOLO_MODELS:
        model_inst = _GLOBAL_YOLO_MODELS[target_name]
        return model_inst if model_inst is not False else None

    try:
        from ultralytics import YOLO
        logger.info(f"[YOLO Singleton] Attempting to load YOLO model '{target_name}'...")
        model_inst = YOLO(target_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[YOLO Singleton] Loaded YOLO model '{target_name}' on device '{device}'.")
        _GLOBAL_YOLO_MODELS[target_name] = model_inst
        return model_inst
    except Exception as e:
        logger.warning(f"[YOLO Singleton] Could not load model '{target_name}' ({e}). Attempting fallback to 'yolov8n.pt'...")
        if target_name != "yolov8n.pt":
            try:
                from ultralytics import YOLO
                fallback_inst = YOLO("yolov8n.pt")
                logger.info("[YOLO Singleton] Successfully loaded fallback model 'yolov8n.pt'.")
                _GLOBAL_YOLO_MODELS[target_name] = fallback_inst
                return fallback_inst
            except Exception as fb_err:
                logger.error(f"[YOLO Singleton] Fallback model load failed: {fb_err}")

        _GLOBAL_YOLO_MODELS[target_name] = False
        return None


class YOLOObjectDetector:
    """Centralized YOLO Object Detector with batch inference, high resolution, caching, and threshold control."""

    def __init__(self, model_name: str = None, confidence_threshold: float = None):
        self.model_name = model_name or settings.YOLO_MODEL
        self.conf_thresh = confidence_threshold if confidence_threshold is not None else settings.YOLO_CONFIDENCE
=======

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
>>>>>>> 2c7b29f0fb7a584d5d69fc583961533e27bda09e

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
<<<<<<< HEAD
        """
        Batch YOLO inference for multiple frames.
        Reuses versioned cache and executes batched model.predict() in a single call.
        """
        conf_val = confidence_threshold if confidence_threshold is not None else self.conf_thresh
        iou_val = getattr(settings, "YOLO_IOU_THRESHOLD", 0.50)
        imgsz_val = getattr(settings, "YOLO_IMGSZ", 960)
        max_det_val = getattr(settings, "YOLO_MAX_DET", 300)

=======
        """Run deterministic batched YOLO detection over sampled video frames."""
        conf_val = (
            confidence_threshold
            if confidence_threshold is not None
            else self.conf_thresh
        )
>>>>>>> 2c7b29f0fb7a584d5d69fc583961533e27bda09e
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
                model_name=f"{self.model_name}_conf{conf_val:.2f}_iou{iou_val:.2f}_sz{imgsz_val}",
            )
            cached = cache_manager.get(cache_key)
            if cached is not None:
                batch_results[str(path)] = [YOLODetection(**d) for d in cached]
            else:
                uncached_paths.append(path)

        if not uncached_paths:
            return batch_results

        model = self.model
<<<<<<< HEAD
        if model:
            try:
                sources = [str(p) for p in uncached_paths]
                # Run batch prediction with configurable parameters
                results = model.predict(
                    source=sources,
                    conf=conf_val,
                    iou=iou_val,
                    imgsz=imgsz_val,
                    max_det=max_det_val,
                    verbose=False
                )
                
                for path, r in zip(uncached_paths, results):
                    path_str = str(path)
                    detections: List[YOLODetection] = []
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = r.names[cls_id]
                        conf = float(box.conf[0].item())
                        xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]

                        x1, y1, x2, y2 = xyxy
                        w = round(max(0.0, x2 - x1), 1)
                        h = round(max(0.0, y2 - y1), 1)
                        cx = round(x1 + w / 2.0, 1)
                        cy = round(y1 + h / 2.0, 1)
                        area = round(w * h, 1)
=======
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
>>>>>>> 2c7b29f0fb7a584d5d69fc583961533e27bda09e

                        detections.append(
                            YOLODetection(
                                class_name=cls_name,
<<<<<<< HEAD
                                confidence=round(conf, 3),
                                bbox=xyxy,
                                center_x=cx,
                                center_y=cy,
                                width=w,
                                height=h,
                                area=area,
=======
                                confidence=round(confidence, 3),
                                bbox=xyxy,
>>>>>>> 2c7b29f0fb7a584d5d69fc583961533e27bda09e
                            )
                        )

<<<<<<< HEAD
                    # Cache detection result
                    cache_key = cache_manager.build_versioned_key(
                        prefix=f"yolo_det_{path.name}",
                        video_hash=video_hash,
                        model_name=f"{self.model_name}_conf{conf_val:.2f}_iou{iou_val:.2f}_sz{imgsz_val}",
                    )
                    cache_manager.set(cache_key, [d.model_dump() for d in detections])
                    logger.info(f"YOLO inference on {path.name} (conf={conf_val:.2f}, imgsz={imgsz_val}): {len(detections)} accepted detections")
=======
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
>>>>>>> 2c7b29f0fb7a584d5d69fc583961533e27bda09e

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

<<<<<<< HEAD
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
                            center_x=200.0,
                            center_y=310.0,
                            width=200.0,
                            height=380.0,
                            area=76000.0,
                        ),
                        YOLODetection(
                            class_name="backpack",
                            confidence=0.87,
                            bbox=[250.0, 350.0, 380.0, 480.0],
                            center_x=315.0,
                            center_y=415.0,
                            width=130.0,
                            height=130.0,
                            area=16900.0,
                        ),
                    ]
=======
        except Exception as e:
            logger.exception(f"YOLO batch detection error: {e}")
            for path in uncached_paths:
                batch_results.setdefault(str(path), [])
>>>>>>> 2c7b29f0fb7a584d5d69fc583961533e27bda09e

        return batch_results


object_detector = YOLOObjectDetector()
