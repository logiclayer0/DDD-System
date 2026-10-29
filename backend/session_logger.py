"""CSV-backed persistence and summaries for detection sessions."""

import csv
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
FIELDNAMES = (
    "timestamp",
    "ear_value",
    "status",
    "is_drowsy",
    "event_type",
    "session_id",
    "session_start_time",
    "session_end_time",
    "duration_seconds",
)


class SessionLogger:
    """Record detection events and alerts to a per-session CSV file."""

    def __init__(self, log_dir: str | Path = "data/session_logs/") -> None:
        """Initialize the session directory and empty active-session state."""
        self.log_dir = Path(log_dir)
        if not self.log_dir.is_absolute():
            self.log_dir = Path(__file__).resolve().parents[1] / self.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id: str | None = None
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.events: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def start_session(self) -> str:
        """Create and return a timestamp-based unique session identifier."""
        with self._lock:
            self.start_time = datetime.now(timezone.utc)
            self.end_time = None
            self.session_id = self.start_time.strftime("%Y%m%dT%H%M%S%fZ")
            self.events = []
            LOGGER.info("Started session %s", self.session_id)
            return self.session_id

    def _append_event(
        self,
        ear_value: float | None,
        status: str,
        is_drowsy: bool,
        event_type: str,
    ) -> None:
        if self.session_id is None:
            raise RuntimeError("Start a session before logging events.")
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ear_value": "" if ear_value is None else float(ear_value),
            "status": status,
            "is_drowsy": bool(is_drowsy),
            "event_type": event_type,
        }
        self.events.append(event)

    def log_event(
        self,
        ear_value: float | None,
        status: str,
        is_drowsy: bool,
    ) -> None:
        """Append one frame result to the active session."""
        with self._lock:
            self._append_event(ear_value, status, is_drowsy, "frame")

    def log_alert(self) -> None:
        """Append an alert event to the active session."""
        with self._lock:
            self._append_event(None, "DROWSY!", True, "alert")

    def _write_csv(self) -> Path:
        if self.session_id is None:
            raise RuntimeError("No active session is available to export.")
        path = self.log_dir / f"{self.session_id}.csv"
        end_time = self.end_time or datetime.now(timezone.utc)
        duration_seconds = (
            (end_time - self.start_time).total_seconds()
            if self.start_time is not None
            else 0.0
        )
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(
                {
                    **event,
                    "session_id": self.session_id,
                    "session_start_time": self.start_time.isoformat(),
                    "session_end_time": end_time.isoformat(),
                    "duration_seconds": duration_seconds,
                }
                for event in self.events
            )
        return path

    def end_session(self) -> dict[str, Any]:
        """End the session, save its CSV, and return summary statistics."""
        with self._lock:
            if self.session_id is None or self.start_time is None:
                raise RuntimeError("No active session is available to end.")
            self.end_time = datetime.now(timezone.utc)
            path = self._write_csv()
            frame_events = [
                event
                for event in self.events
                if event["event_type"] == "frame"
            ]
            ear_values = [
                float(event["ear_value"])
                for event in frame_events
                if event["ear_value"] != ""
            ]
            alert_count = sum(
                event["event_type"] == "alert" for event in self.events
            )
            drowsy_frames = sum(
                bool(event["is_drowsy"]) for event in frame_events
            )
            summary = {
                "session_id": self.session_id,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration_seconds": (
                    self.end_time - self.start_time
                ).total_seconds(),
                "total_alerts": int(alert_count),
                "drowsy_frames": int(drowsy_frames),
                "frame_count": len(frame_events),
                "average_ear": (
                    sum(ear_values) / len(ear_values) if ear_values else None
                ),
                "minimum_ear": min(ear_values) if ear_values else None,
                "maximum_ear": max(ear_values) if ear_values else None,
                "csv_path": str(path),
            }
            LOGGER.info("Ended session %s", self.session_id)
            return summary

    def export_to_csv(self, session_id: str) -> Path:
        """Return an existing session CSV or export the active session."""
        if not session_id or Path(session_id).name != session_id:
            raise ValueError("session_id must be a plain session identifier.")
        path = self.log_dir / f"{session_id}.csv"
        with self._lock:
            if session_id == self.session_id:
                return self._write_csv()
        if not path.is_file():
            raise FileNotFoundError(f"Session CSV was not found: {session_id}")
        return path
