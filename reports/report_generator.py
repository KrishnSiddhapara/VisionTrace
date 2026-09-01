from pathlib import Path
from typing import Dict, Any

from models.schemas import VideoMemory
from reports.json_export import json_exporter
from reports.pdf_export import pdf_exporter
from utils.logger import logger

class ReportGenerator:
    """Master report generation orchestrator."""

    def generate_all_reports(self, memory: VideoMemory) -> Dict[str, Path]:
        logger.info(f"Generating JSON, CSV, and PDF reports for {memory.metadata.filename}...")
        json_path = json_exporter.export_json(memory)
        csv_path = json_exporter.export_csv(memory)
        pdf_path = pdf_exporter.export_pdf(memory)

        return {
            "json": json_path,
            "csv": csv_path,
            "pdf": pdf_path,
        }

report_generator = ReportGenerator()
