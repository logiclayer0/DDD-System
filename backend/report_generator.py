"""Professional PDF report generation for recorded detection sessions."""

import logging
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

LOGGER = logging.getLogger(__name__)
NAVY = colors.HexColor("#0A192F")
CYAN = colors.HexColor("#64FFDA")


class ReportGenerator:
    """Build a formatted PDF report and EAR timeline for one session."""

    def __init__(
        self,
        session_data: dict[str, Any],
        output_dir: str | Path = "reports/",
    ) -> None:
        """Store session data and prepare the report output directory."""
        self.session_data = session_data
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = (
                Path(__file__).resolve().parents[1] / self.output_dir
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _events_frame(self) -> pd.DataFrame:
        events = self.session_data.get("events", [])
        frame = (
            events.copy()
            if isinstance(events, pd.DataFrame)
            else pd.DataFrame(events)
        )
        for column in (
            "timestamp",
            "ear_value",
            "status",
            "is_drowsy",
            "event_type",
        ):
            if column not in frame.columns:
                frame[column] = pd.Series(dtype="object")
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], errors="coerce", utc=True
        )
        frame["ear_value"] = pd.to_numeric(frame["ear_value"], errors="coerce")
        frame["is_drowsy"] = (
            frame["is_drowsy"]
            .astype(str)
            .str.lower()
            .isin(("true", "1", "yes"))
        )
        return frame

    def _timeline_image(self, events: pd.DataFrame) -> BytesIO:
        frame_events = events[
            events["event_type"].eq("frame") & events["ear_value"].notna()
        ]
        figure, axis = plt.subplots(figsize=(8, 3))
        if not frame_events.empty:
            axis.plot(
                frame_events["timestamp"],
                frame_events["ear_value"],
                color="#0A192F",
                linewidth=1.6,
            )
            drowsy_events = frame_events[frame_events["is_drowsy"]]
            if not drowsy_events.empty:
                axis.scatter(
                    drowsy_events["timestamp"],
                    drowsy_events["ear_value"],
                    color="#D64545",
                    s=18,
                    label="Drowsy",
                )
                axis.legend(loc="best")
        axis.set_title("Eye Aspect Ratio Timeline", color="#0A192F")
        axis.set_xlabel("Time")
        axis.set_ylabel("EAR")
        axis.grid(alpha=0.2)
        figure.autofmt_xdate()
        figure.tight_layout()
        image_buffer = BytesIO()
        figure.savefig(image_buffer, format="png", dpi=150)
        plt.close(figure)
        image_buffer.seek(0)
        return image_buffer

    def generate_pdf(self) -> Path:
        """Generate and return the session PDF report path."""
        session_id = str(self.session_data.get("session_id", "session"))
        if Path(session_id).name != session_id:
            raise ValueError("session_id must be a plain session identifier.")
        events = self._events_frame()
        frame_events = events[events["event_type"].eq("frame")]
        ear_values = frame_events["ear_value"].dropna()
        alerts = events[events["event_type"].eq("alert")]
        drowsy_events = frame_events[frame_events["is_drowsy"]]
        start_time = self.session_data.get("start_time", "Unavailable")
        end_time = self.session_data.get("end_time", "Unavailable")
        duration = self.session_data.get("duration_seconds")
        if (
            duration is None
            and start_time != "Unavailable"
            and end_time != "Unavailable"
        ):
            start_value = pd.to_datetime(start_time, utc=True, errors="coerce")
            end_value = pd.to_datetime(end_time, utc=True, errors="coerce")
            duration = (
                (end_value - start_value).total_seconds()
                if pd.notna(start_value) and pd.notna(end_value)
                else 0
            )
        summary = [
            ["Session Summary", "Value"],
            ["Start time", str(start_time)],
            ["End time", str(end_time)],
            ["Duration", f"{float(duration or 0):.1f} seconds"],
            ["Total alerts", str(len(alerts))],
        ]
        statistics = [
            ["Statistic", "Value"],
            [
                "Average EAR",
                f"{ear_values.mean():.4f}" if not ear_values.empty else "N/A",
            ],
            [
                "Minimum EAR",
                f"{ear_values.min():.4f}" if not ear_values.empty else "N/A",
            ],
            [
                "Maximum EAR",
                f"{ear_values.max():.4f}" if not ear_values.empty else "N/A",
            ],
            ["Drowsy frame count", str(int(drowsy_events.shape[0]))],
            ["Alert count", str(len(alerts))],
        ]
        destination = self.output_dir / f"{session_id}.pdf"
        document = SimpleDocTemplate(
            str(destination),
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=20 * mm,
            title=f"Driver Drowsiness Detection Report - {session_id}",
            author="Driver Drowsiness Detection System",
        )
        styles = getSampleStyleSheet()
        styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=styles["Title"],
                textColor=NAVY,
                alignment=TA_CENTER,
                fontSize=20,
                leading=25,
                spaceAfter=8,
            )
        )
        styles.add(
            ParagraphStyle(
                name="SectionHeading",
                parent=styles["Heading2"],
                textColor=NAVY,
                spaceBefore=12,
                spaceAfter=6,
            )
        )
        story: list[Any] = [
            Paragraph(
                "Driver Drowsiness Detection System", styles["ReportTitle"]
            ),
            Paragraph("Session Analysis Report", styles["Heading2"]),
            Paragraph(
                "Generated "
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
                f"| Session ID: {session_id}",
                styles["Normal"],
            ),
            Spacer(1, 8),
        ]
        logo_path = Path(__file__).resolve().parents[1] / "assets" / "logo.png"
        if logo_path.is_file():
            story.append(Image(str(logo_path), width=28 * mm, height=28 * mm))
        story.extend(
            [
                Paragraph("Session Summary", styles["SectionHeading"]),
                self._styled_table(summary),
                Paragraph("Statistics", styles["SectionHeading"]),
                self._styled_table(statistics),
                Paragraph("EAR Timeline", styles["SectionHeading"]),
                Image(
                    self._timeline_image(events),
                    width=165 * mm,
                    height=62 * mm,
                ),
                PageBreak(),
                Paragraph("Drowsiness Events Log", styles["SectionHeading"]),
            ]
        )
        event_rows = [["Timestamp", "EAR", "Status", "Event"]]
        for _, event in drowsy_events.iterrows():
            event_rows.append(
                [
                    (
                        event["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
                        if pd.notna(event["timestamp"])
                        else "Unavailable"
                    ),
                    f"{event['ear_value']:.4f}",
                    str(event["status"]),
                    "Drowsy frame",
                ]
            )
        for _, event in alerts.iterrows():
            event_rows.append(
                [
                    (
                        event["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
                        if pd.notna(event["timestamp"])
                        else "Unavailable"
                    ),
                    "N/A",
                    str(event["status"]),
                    "Alert",
                ]
            )
        event_rows[1:] = sorted(event_rows[1:], key=lambda row: row[0])
        if len(event_rows) == 1:
            event_rows.append(["No drowsiness events", "", "", ""])
        story.append(
            self._styled_table(
                event_rows, [76 * mm, 22 * mm, 30 * mm, 32 * mm]
            )
        )
        document.build(
            story,
            onFirstPage=self._draw_footer,
            onLaterPages=self._draw_footer,
        )
        LOGGER.info("Generated session report at %s", destination)
        return destination

    @staticmethod
    def _styled_table(
        rows: list[list[str]],
        widths: list[float] | None = None,
    ) -> Table:
        table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.35,
                        colors.HexColor("#D7DEE8"),
                    ),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#F2F5F8")],
                    ),
                    ("TEXTCOLOR", (0, 1), (-1, -1), NAVY),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("LEADING", (0, 0), (-1, -1), 11),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return table

    @staticmethod
    def _draw_footer(canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(CYAN)
        canvas.setLineWidth(1)
        canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(NAVY)
        canvas.drawString(
            18 * mm,
            10 * mm,
            "Generated "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        )
        canvas.drawRightString(192 * mm, 10 * mm, f"Page {document.page}")
        canvas.restoreState()
