from typing import List, Dict, Any
from models.schemas import FrameObservation, TrackedObject
from utils.logger import logger

class AnomalyDetector:
    """Flags unusual visual changes (sudden movement, unexpected object appearance, etc.)."""

    def detect_anomalies(self, frame_observations: List[FrameObservation], tracks: List[TrackedObject]) -> List[Dict[str, Any]]:
        anomalies = []

        for obs in frame_observations:
            for unc in obs.uncertainties:
                anomalies.append({
                    "timestamp": obs.timestamp,
                    "type": "Visual Ambiguity / Uncertainty",
                    "description": unc,
                    "confidence": 0.65,
                })

        logger.info(f"Detected {len(anomalies)} visual anomalies/uncertainties.")
        return anomalies

anomaly_detector = AnomalyDetector()
