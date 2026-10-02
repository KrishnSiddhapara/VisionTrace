from pathlib import Path
from typing import List, Tuple, Union, Optional
import cv2
import numpy as np

from config.settings import settings
from utils.logger import logger


class PersonVisualGrounder:
    """
    Generates person-specific two-level visual evidence:
    Evidence A: Cropped image around target person's bounding box with padding.
    Evidence B: Full frame image with target person visually highlighted (emerald box + banner).
    """

    def generate_person_evidence_images(
        self,
        image_path: Union[str, Path],
        bbox: List[float],
        person_id: str,
        out_dir: Union[str, Path]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate (crop_path, highlight_path) for target person.
        """
        src_path = Path(image_path).resolve()
        if not src_path.exists() or not bbox or len(bbox) < 4:
            return None, None

        out_path = Path(out_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        clean_p_id = person_id.replace("#", "").replace(" ", "_").lower()
        frame_name = src_path.stem

        crop_file = out_path / f"evidenceA_{clean_p_id}_{frame_name}.jpg"
        highlight_file = out_path / f"evidenceB_{clean_p_id}_{frame_name}.jpg"

        # Return cached paths if already generated
        if crop_file.exists() and highlight_file.exists():
            return str(crop_file), str(highlight_file)

        img = cv2.imread(str(src_path))
        if img is None or img.size == 0:
            return None, None

        img_h, img_w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        # 1. Evidence A: Crop with 20% padding around person bbox
        pad_w = 0.20 * w
        pad_h = 0.20 * h
        crop_x1 = max(0, int(x1 - pad_w))
        crop_y1 = max(0, int(y1 - pad_h))
        crop_x2 = min(img_w, int(x2 + pad_w))
        crop_y2 = min(img_h, int(y2 + pad_h))

        crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop_img.size > 0:
            cv2.imwrite(str(crop_file), crop_img)
        else:
            cv2.imwrite(str(crop_file), img)

        # 2. Evidence B: Full frame with target person highlighted in bright emerald BGR (102, 255, 0)
        highlight_img = img.copy()
        box_x1, box_y1 = max(0, int(x1)), max(0, int(y1))
        box_x2, box_y2 = min(img_w, int(x2)), min(img_h, int(y2))

        # Thick 3px green outline
        cv2.rectangle(highlight_img, (box_x1, box_y1), (box_x2, box_y2), (102, 255, 0), 3)

        # Label banner
        label_str = f"TARGET PERSON: {person_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (lbl_w, lbl_h), baseline = cv2.getTextSize(label_str, font, scale, thickness)

        bg_y1 = max(0, box_y1 - lbl_h - 6)
        bg_y2 = box_y1
        cv2.rectangle(highlight_img, (box_x1, bg_y1), (box_x1 + lbl_w + 10, bg_y2), (102, 255, 0), -1)
        cv2.putText(highlight_img, label_str, (box_x1 + 5, box_y1 - 4), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

        cv2.imwrite(str(highlight_file), highlight_img)

        return str(crop_file), str(highlight_file)


person_visual_grounder = PersonVisualGrounder()
