from pathlib import Path
from typing import List, Union, Dict, Any, Optional
import torch

from config.settings import settings
from models.schemas import YOLODetection
from utils.logger import logger
from utils.caching import cache_manager

_GLOBAL_YOLO_MODELS: Dict[str, Any] = {}


def get_shared_yolo_model(model_name: Optional[str] = None):
    """Load YOLO model instance ONLY ONCE across the application runtime with fallback support."""
    global _GLOBAL_YOLO_MODELS
    target_name = model_name or settings.YOLO_MODEL

    if target_name in _GLOBAL_YOLO_MODELS:
        model_inst = _GLOBAL_YOLO_MODELS[target_name]
        return model_inst if model_inst is not False else None

    try:
        from ultralytics import YOLO
        logger.info(f"[YOLO Singleton] Loading YOLO model '{target_name}'...")
        model_inst = YOLO(target_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[YOLO Singleton] Successfully loaded '{target_name}' on device '{device}'.")
        _GLOBAL_YOLO_MODELS[target_name] = model_inst
        return model_inst
    except Exception as e:
        logger.warning(f"[YOLO Singleton] Could not load model '{target_name}' ({e}). Fallback to 'yolov8n.pt'...")
        if target_name != "yolov8n.pt":
            try:
                from ultralytics import YOLO
                fallback_inst = YOLO("yolov8n.pt")
                logger.info("[YOLO Singleton] Loaded fallback model 'yolov8n.pt'.")
                _GLOBAL_YOLO_MODELS[target_name] = fallback_inst
                return fallback_inst
            except Exception as fb_err:
                logger.error(f"[YOLO Singleton] Fallback model load failed: {fb_err}")

        _GLOBAL_YOLO_MODELS[target_name] = False
        return None


class YOLOObjectDetector:
    """Centralized high-accuracy YOLO Object Detector with intra-frame NMS, batch inference, and versioned caching."""

    def __init__(self, model_name: Optional[str] = None, confidence_threshold: Optional[float] = None):
        self.model_name = model_name or settings.YOLO_MODEL
        self.conf_thresh = confidence_threshold if confidence_threshold is not None else settings.YOLO_CONFIDENCE

    @property
    def model(self):
        return get_shared_yolo_model(self.model_name)

    def suppress_duplicate_person_detections(self, detections: List[YOLODetection]) -> List[YOLODetection]:
        """
        Apply person-specific intra-frame NMS and duplicate suppression.
        If multiple boxes represent the same physical person in a single frame, suppress lower-confidence box.
        """
        person_dets = [d for d in detections if d.class_name.lower() in ("person", "people", "human", "man", "woman", "child", "kid", "player")]
        other_dets = [d for d in detections if d not in person_dets]

        if len(person_dets) <= 1:
            return detections

        # Sort person detections by confidence descending
        sorted_persons = sorted(person_dets, key=lambda d: d.confidence, reverse=True)
        kept_persons: List[YOLODetection] = []

        from vision.tracker import calculate_bbox_iou, calculate_center_distance

        for p_det in sorted_persons:
            is_duplicate = False
            for kept in kept_persons:
                iou = calculate_bbox_iou(p_det.bbox, kept.bbox)
                dist = calculate_center_distance(p_det.bbox, kept.bbox)

                w_kept = max(1.0, kept.width)
                h_kept = max(1.0, kept.height)
                w_p = max(1.0, p_det.width)
                h_p = max(1.0, p_det.height)
                area_ratio = max(w_p * h_p / (w_kept * h_kept), (w_kept * h_kept) / (w_p * h_p))

                # If high overlap (IoU >= NMS threshold) OR very close center distance with similar size -> duplicate
                nms_iou_thresh = getattr(settings, "PERSON_NMS_IOU", 0.40)
                if iou >= nms_iou_thresh or (dist < 25.0 and area_ratio < 2.0):
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept_persons.append(p_det)

        return kept_persons + other_dets

    def detect_objects(
        self,
        image_path: Union[str, Path],
        video_hash: str = "",
        confidence_threshold: Optional[float] = None,
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
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, List[YOLODetection]]:
        """
        Batch YOLO inference for multiple frames.
        Reuses versioned cache and executes batched model.predict() in a single call.
        """
        conf_val = confidence_threshold if confidence_threshold is not None else self.conf_thresh
        person_conf_val = getattr(settings, "PERSON_CONFIDENCE", 0.55)
        iou_val = getattr(settings, "YOLO_IOU_THRESHOLD", 0.50)
        imgsz_val = getattr(settings, "YOLO_IMGSZ", 960)
        max_det_val = getattr(settings, "YOLO_MAX_DET", 300)
        trk_ver = getattr(settings, "TRACKING_VERSION", "v3.2")

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
                model_name=f"{self.model_name}_conf{conf_val:.2f}_pconf{person_conf_val:.2f}_iou{iou_val:.2f}_sz{imgsz_val}_{trk_ver}",
            )
            cached = cache_manager.get(cache_key)
            if cached is not None:
                batch_results[str(path)] = [YOLODetection(**d) for d in cached]
            else:
                uncached_paths.append(path)

        if not uncached_paths:
            return batch_results

        model = self.model
        if model is not None:
            try:
                sources = [str(p) for p in uncached_paths]
                device = "cuda" if torch.cuda.is_available() else "cpu"
                # Use min threshold to capture raw detections, then apply person-specific filter
                effective_conf = min(conf_val, person_conf_val)

                results = model.predict(
                    source=sources,
                    conf=effective_conf,
                    iou=iou_val,
                    imgsz=imgsz_val,
                    max_det=max_det_val,
                    device=device,
                    verbose=False,
                )

                for path, r in zip(uncached_paths, results):
                    path_str = str(path)
                    raw_detections: List[YOLODetection] = []
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            cls_name = r.names[cls_id]
                            conf = float(box.conf[0].item())

                            # Filter based on class-specific confidence threshold
                            is_person = cls_name.lower() in ("person", "people", "human", "man", "woman", "child", "kid", "player")
                            required_conf = person_conf_val if is_person else conf_val

                            if conf < required_conf:
                                continue

                            xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
                            x1, y1, x2, y2 = xyxy
                            w = round(max(0.0, x2 - x1), 1)
                            h = round(max(0.0, y2 - y1), 1)
                            cx = round(x1 + w / 2.0, 1)
                            cy = round(y1 + h / 2.0, 1)
                            area = round(w * h, 1)

                            raw_detections.append(
                                YOLODetection(
                                    class_name=cls_name,
                                    confidence=round(conf, 3),
                                    bbox=xyxy,
                                    center_x=cx,
                                    center_y=cy,
                                    width=w,
                                    height=h,
                                    area=area,
                                )
                            )

                    # Intra-frame NMS / duplicate suppression for person boxes
                    deduped_detections = self.suppress_duplicate_person_detections(raw_detections)

                    # Stable deterministic ordering
                    deduped_detections.sort(
                        key=lambda d: (d.class_name, d.bbox[0], d.bbox[1], -d.confidence)
                    )
                    batch_results[path_str] = deduped_detections

                    cache_key = cache_manager.build_versioned_key(
                        prefix=f"yolo_det_{path.name}",
                        video_hash=video_hash,
                        model_name=f"{self.model_name}_conf{conf_val:.2f}_pconf{person_conf_val:.2f}_iou{iou_val:.2f}_sz{imgsz_val}_{trk_ver}",
                    )
                    cache_manager.set(cache_key, [d.model_dump() for d in deduped_detections])
                    logger.info(
                        f"YOLO: {path.name}: {len(deduped_detections)} accepted detections "
                        f"(person_conf={person_conf_val:.2f}, conf={conf_val:.2f}, imgsz={imgsz_val})"
                    )

            except Exception as e:
                logger.exception(f"YOLO batch detection error: {e}")
                for p in uncached_paths:
                    batch_results.setdefault(str(p), [])

        # Fallback mock detection ONLY when explicit VLM_MOCK_MODE is True and no results
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

        return batch_results


object_detector = YOLOObjectDetector()
