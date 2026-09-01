from pathlib import Path
from typing import Union

from config.settings import settings
from models.schemas import VideoMemory
from utils.logger import logger

class PDFExporter:
    """Generates PDF visual video understanding report."""

    def export_pdf(self, memory: VideoMemory, output_path: Union[str, Path] = None) -> Path:
        out = Path(output_path) if output_path else settings.OUTPUTS_DIR / f"report_{memory.video_hash}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors

            doc = SimpleDocTemplate(str(out), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            title_style = ParagraphStyle(
                'ReportTitle',
                parent=styles['Heading1'],
                fontSize=20,
                textColor=colors.HexColor('#6366f1'),
                spaceAfter=12
            )

            story.append(Paragraph("🎥 VisionTrace AI - Video Analysis Report", title_style))
            story.append(Spacer(1, 10))

            meta = memory.metadata
            meta_text = (
                f"<b>Filename:</b> {meta.filename}<br/>"
                f"<b>Duration:</b> {meta.duration_sec}s | <b>FPS:</b> {meta.fps} | <b>Resolution:</b> {meta.resolution_str}<br/>"
                f"<b>Video Hash:</b> {meta.video_hash[:16]}"
            )
            story.append(Paragraph(meta_text, styles['Normal']))
            story.append(Spacer(1, 15))

            # Executive Summary
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            summary_txt = memory.summary.get("standard", "Summary unavailable.")
            story.append(Paragraph(summary_txt, styles['Normal']))
            story.append(Spacer(1, 15))

            # Timeline
            story.append(Paragraph("Chronological Timeline", styles['Heading2']))
            table_data = [["Step", "Timestamp", "Type", "Event Description"]]
            for step in memory.timeline[:15]:
                table_data.append([
                    str(step.get("step_index", "")),
                    step.get("formatted_time", ""),
                    step.get("event_type", ""),
                    step.get("description", "")[:60]
                ])

            t = Table(table_data, colWidths=[40, 70, 80, 280])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
            ]))
            story.append(t)

            doc.build(story)
            logger.info(f"Generated PDF report: {out}")
        except Exception as e:
            logger.warning(f"ReportLab PDF export fallback: {e}")
            # Text fallback file if reportlab has an error
            with open(out.with_suffix(".txt"), "w", encoding="utf-8") as f:
                f.write(f"VISIONTRACE AI REPORT - {memory.metadata.filename}\n")
                f.write(memory.summary.get("standard", ""))
            return out.with_suffix(".txt")

        return out

pdf_exporter = PDFExporter()
