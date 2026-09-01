import json
from pathlib import Path
from typing import Union
import pandas as pd

from config.settings import settings
from models.schemas import VideoMemory

class JSONExporter:
    """Exports video memory to JSON and CSV formats."""

    def export_json(self, memory: VideoMemory, output_path: Union[str, Path] = None) -> Path:
        out = Path(output_path) if output_path else settings.OUTPUTS_DIR / f"report_{memory.video_hash}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(memory.model_dump_json(indent=2))
        return out

    def export_csv(self, memory: VideoMemory, output_path: Union[str, Path] = None) -> Path:
        out = Path(output_path) if output_path else settings.OUTPUTS_DIR / f"events_{memory.video_hash}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for evt in memory.events:
            rows.append({
                "event_id": evt.event_id,
                "start_time": evt.start_time,
                "end_time": evt.end_time,
                "event_type": evt.event_type,
                "subject": evt.subject or "",
                "object": evt.object or "",
                "description": evt.description,
                "confidence": evt.confidence,
            })

        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)
        return out

json_exporter = JSONExporter()
