import os
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from utils.logger import logger

@dataclass
class StageMetric:
    stage_name: str
    start_time: float
    end_time: float = 0.0
    duration: float = 0.0
    item_count: int = 0
    unit: str = "frames"
    details: Dict[str, Any] = field(default_factory=dict)

class PerformanceProfiler:
    """
    Lightweight, thread-safe performance profiler for VisionTrace AI execution pipeline.
    Tracks start time, end time, duration, and processed counts for each stage.
    """

    def __init__(self):
        self.metrics: Dict[str, StageMetric] = {}
        self.active_stages: Dict[str, float] = {}
        self.pipeline_start_time: Optional[float] = None
        self.pipeline_end_time: Optional[float] = None
        self.debug_mode: bool = os.getenv("VISIONTRACE_DEBUG_PERFORMANCE", "False").lower() in ("true", "1", "yes")

    def start_pipeline(self) -> None:
        self.metrics.clear()
        self.active_stages.clear()
        self.pipeline_start_time = time.perf_counter()
        if self.debug_mode:
            logger.info("[PERF] === Started VisionTrace AI Video Processing Pipeline ===")

    def start_stage(self, stage_name: str) -> None:
        self.active_stages[stage_name] = time.perf_counter()

    def stop_stage(self, stage_name: str, item_count: int = 0, unit: str = "frames", **details) -> float:
        start_t = self.active_stages.pop(stage_name, None)
        if start_t is None:
            return 0.0
        
        end_t = time.perf_counter()
        duration = round(end_t - start_t, 3)

        metric = StageMetric(
            stage_name=stage_name,
            start_time=start_t,
            end_time=end_t,
            duration=duration,
            item_count=item_count,
            unit=unit,
            details=details
        )
        self.metrics[stage_name] = metric

        count_str = f" | {item_count} {unit}" if item_count > 0 else ""
        log_msg = f"[PERF] {stage_name}: {duration:.2f} sec{count_str}"
        logger.info(log_msg)

        return duration

    def stop_pipeline(self) -> float:
        if self.pipeline_start_time is None:
            return 0.0
        self.pipeline_end_time = time.perf_counter()
        total_duration = round(self.pipeline_end_time - self.pipeline_start_time, 3)
        logger.info(f"[PERF] Pipeline Completed Total Duration: {total_duration:.2f} sec across {len(self.metrics)} stages.")
        return total_duration

    def get_summary(self) -> Dict[str, Any]:
        total_duration = round(
            (self.pipeline_end_time - self.pipeline_start_time) if self.pipeline_start_time and self.pipeline_end_time else 0.0,
            2
        )
        stage_breakdown = {}
        for name, m in self.metrics.items():
            stage_breakdown[name] = {
                "duration_sec": m.duration,
                "item_count": m.item_count,
                "unit": m.unit,
                "details": m.details
            }
        return {
            "total_duration_sec": total_duration,
            "stages": stage_breakdown
        }

profiler = PerformanceProfiler()
