"""Streamlit app for driver drowsiness detection and analytics."""

import hashlib
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from backend.alert_system import AlertSystem
from backend.detector import DrowsinessDetector
from backend.report_generator import ReportGenerator
from backend.session_logger import SessionLogger
from frontend.charts import drowsiness_timeline, ear_histogram, ear_timeline
from frontend.components import (
    info_card,
    metric_card,
    section_header,
    status_banner,
)
from frontend.styles import APP_CSS

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "data" / "session_logs"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = ROOT / "models" / "shape_predictor_68_face_landmarks.dat"
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    page_icon="🚗",
    layout="wide",
)
st.markdown(APP_CSS, unsafe_allow_html=True)


def get_detector() -> DrowsinessDetector:
    """Create and store one detector per browser session."""
    if "detector" not in st.session_state:
        detector = DrowsinessDetector(model_path=MODEL_PATH)
        detector.load_models()
        st.session_state.detector = detector
    return st.session_state.detector


def get_logger() -> SessionLogger:
    """Return the session logger stored for the current browser session."""
    if "session_logger" not in st.session_state:
        st.session_state.session_logger = SessionLogger(LOG_DIR)
    return st.session_state.session_logger


def get_alert_system() -> AlertSystem:
    """Return the session-scoped alarm player."""
    if "alert_system" not in st.session_state:
        st.session_state.alert_system = AlertSystem(
            ROOT / "assets" / "alarm.wav"
        )
    return st.session_state.alert_system


def load_session_csv(session_id: str) -> pd.DataFrame:
    """Load one validated session CSV from the application log directory."""
    if not session_id or Path(session_id).name != session_id:
        raise ValueError("Invalid session identifier.")
    csv_path = LOG_DIR / f"{session_id}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"No CSV file found for session {session_id}.")
    return pd.read_csv(csv_path)


def list_sessions() -> list[str]:
    """Return available session IDs from persisted CSV files."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        (path.stem for path in LOG_DIR.glob("*.csv")),
        reverse=True,
    )


def render_home(detector: DrowsinessDetector) -> None:
    """Render the home dashboard and model status."""
    st.markdown(
        '<div class="hero"><h1>Driver Drowsiness Detection System</h1>'
        "<p>Real-time eye-closure monitoring, timely alerts, analytics, "
        "and exportable reports to support safer driving.</p></div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(3)
    descriptions = [
        (
            "Real-time Detection",
            "Estimate eye openness from facial landmarks and track prolonged "
            "closure.",
        ),
        (
            "Alert System",
            "Trigger a configurable alarm when sustained eye closure is "
            "detected.",
        ),
        (
            "Analytics",
            "Review session trends, event logs, and downloadable PDF reports.",
        ),
    ]
    for column, (title, body) in zip(columns, descriptions):
        with column:
            st.markdown(info_card(title, body), unsafe_allow_html=True)
    model_status = "Loaded" if detector.models_loaded else "Not loaded"
    st.markdown(
        section_header("System Status", "Model and runtime readiness"),
        unsafe_allow_html=True,
    )
    st.markdown(
        status_banner(
            "Models loaded" if detector.models_loaded else "Models not loaded",
            (
                "68-point facial landmark predictor is ready."
                if detector.models_loaded
                else f"Place the predictor file at {MODEL_PATH}."
            ),
        ),
        unsafe_allow_html=True,
    )
    st.caption(f"Predictor status: {model_status}")


def draw_overlay(frame: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    """Draw face bounds and eye landmarks over a BGR frame."""
    output = frame.copy()
    face = result.get("face")
    if face is not None:
        left, top, right, bottom = face
        color = (0, 0, 220) if result.get("is_drowsy") else (40, 170, 80)
        cv2.rectangle(output, (left, top), (right, bottom), color, 2)
    landmarks = result.get("landmarks")
    if landmarks is not None:
        for index in (*range(36, 42), *range(42, 48)):
            point = tuple(int(value) for value in landmarks[index])
            cv2.circle(output, point, 2, (100, 255, 218), -1)
    return output


def render_live_detection(detector: DrowsinessDetector) -> None:
    """Render the camera flow, detector settings, and session controls."""
    st.markdown(
        section_header(
            "Live Detection",
            "Capture a frame to analyze the current driver view.",
        ),
        unsafe_allow_html=True,
    )
    threshold = st.sidebar.slider(
        "EAR Threshold",
        min_value=0.15,
        max_value=0.35,
        value=0.25,
        step=0.01,
        key="ear_threshold",
    )
    consecutive_frames = st.sidebar.slider(
        "Consecutive Frames",
        min_value=10,
        max_value=40,
        value=20,
        step=1,
        key="consecutive_frames",
    )
    alarm_enabled = st.sidebar.toggle(
        "Alarm Sound", value=True, key="alarm_enabled"
    )
    detector.ear_threshold = threshold
    detector.consecutive_frames = consecutive_frames
    logger = get_logger()
    alert_system = get_alert_system()
    active = bool(st.session_state.get("detection_active", False))
    control_columns = st.columns([1, 1, 4])
    with control_columns[0]:
        if st.button(
            "Stop Detection" if active else "Start Detection", type="primary"
        ):
            if active:
                try:
                    summary = logger.end_session()
                    st.session_state.last_session_summary = summary
                    st.success("Session saved.")
                except (OSError, RuntimeError) as error:
                    LOGGER.exception("Unable to save the detection session")
                    st.error(str(error))
                alert_system.stop_alarm()
                detector.reset_state()
                st.session_state.last_drowsy_state = False
                st.session_state.detection_active = False
            else:
                logger.start_session()
                detector.reset_state()
                st.session_state.last_drowsy_state = False
                st.session_state.last_processed_image = None
                st.session_state.camera_generation = (
                    st.session_state.get("camera_generation", 0) + 1
                )
                st.session_state.detection_active = True
            st.rerun()
    with control_columns[1]:
        if active:
            st.caption("Session recording")
    if not detector.models_loaded:
        st.warning(
            "The landmark model is unavailable. Detection requires "
            "`models/shape_predictor_68_face_landmarks.dat`."
        )
    camera_generation = st.session_state.get("camera_generation", 0)
    camera_image = st.camera_input(
        "Capture a frame",
        disabled=not active,
        key=f"camera_input_{camera_generation}",
    )
    if camera_image is None:
        st.info("Start detection, then capture a frame from the camera.")
        return
    if not active:
        st.info("Start detection before analyzing frames.")
        return
    try:
        image_bytes = camera_image.getvalue()
        image_digest = hashlib.sha256(image_bytes).hexdigest()
        if image_digest == st.session_state.get("last_processed_image"):
            st.info("Frame already analyzed. Capture a new image to continue.")
            return
        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise ValueError("The camera image could not be decoded.")
        result = detector.process_frame(image)
        st.session_state.last_processed_image = image_digest
        logger.log_event(result["ear"], result["status"], result["is_drowsy"])
        was_drowsy = bool(st.session_state.get("last_drowsy_state", False))
        if result["is_drowsy"] and not was_drowsy:
            logger.log_alert()
            if alarm_enabled:
                alert_system.play_alarm()
        elif not result["is_drowsy"]:
            alert_system.stop_alarm()
        st.session_state.last_drowsy_state = result["is_drowsy"]
        display_frame = draw_overlay(image, result)
        st.image(
            cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
            caption="Analyzed frame",
        )
        metric_columns = st.columns(4)
        values = (
            (
                "EAR Value",
                f"{result['ear']:.3f}" if result["ear"] is not None else "N/A",
            ),
            ("Status", result["status"]),
            ("Frames Closed", str(result["eyes_closed_frames"])),
            (
                "Alerts Triggered",
                str(
                    sum(
                        event["event_type"] == "alert"
                        for event in logger.events
                    )
                ),
            ),
        )
        for column, (label, value) in zip(metric_columns, values):
            with column:
                st.markdown(metric_card(label, value), unsafe_allow_html=True)
        st.markdown(status_banner(result["status"]), unsafe_allow_html=True)
    except (ValueError, RuntimeError, cv2.error) as error:
        LOGGER.exception("Frame processing failed")
        st.error(f"Could not process this frame: {error}")


def summarize_events(events: pd.DataFrame) -> dict[str, float | int | None]:
    """Calculate summary values from one session event frame."""
    frame_events = (
        events[events["event_type"].eq("frame")].copy()
        if "event_type" in events
        else events.copy()
    )
    frame_events["ear_value"] = pd.to_numeric(
        frame_events.get("ear_value"), errors="coerce"
    )
    ear_values = frame_events["ear_value"].dropna()
    alerts = (
        events[events["event_type"].eq("alert")]
        if "event_type" in events
        else events.iloc[0:0]
    )
    drowsy = (
        frame_events["is_drowsy"]
        .astype(str)
        .str.lower()
        .isin(("true", "1", "yes"))
        if "is_drowsy" in frame_events
        else pd.Series(dtype=bool)
    )
    if (
        "duration_seconds" in events
        and events["duration_seconds"].notna().any()
    ):
        duration = float(
            pd.to_numeric(events["duration_seconds"], errors="coerce")
            .dropna()
            .iloc[0]
        )
    else:
        timestamps = pd.to_datetime(
            events.get("timestamp"), errors="coerce", utc=True
        )
        duration = (
            (timestamps.max() - timestamps.min()).total_seconds()
            if len(timestamps.dropna()) > 1
            else 0.0
        )
    return {
        "duration": duration,
        "alerts": len(alerts),
        "average_ear": (
            float(ear_values.mean()) if not ear_values.empty else None
        ),
        "drowsiness_percent": (
            float(drowsy.mean() * 100) if len(drowsy) else 0.0
        ),
    }


def render_analytics() -> None:
    """Render session selection, KPIs, charts, and event data."""
    st.markdown(
        section_header(
            "Analytics Dashboard", "Explore recorded session performance."
        ),
        unsafe_allow_html=True,
    )
    sessions = list_sessions()
    if not sessions:
        st.info(
            "No saved sessions are available yet. Complete a live "
            "detection session first."
        )
        return
    selected_session = st.selectbox("Session", sessions)
    try:
        events = load_session_csv(selected_session)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        LOGGER.exception("Unable to load selected session")
        st.error(f"Unable to load session: {error}")
        return
    summary = summarize_events(events)
    cards = (
        ("Total Duration", f"{summary['duration']:.1f} s"),
        ("Total Alerts", str(summary["alerts"])),
        (
            "Average EAR",
            (
                f"{summary['average_ear']:.3f}"
                if summary["average_ear"] is not None
                else "N/A"
            ),
        ),
        ("Drowsiness", f"{summary['drowsiness_percent']:.1f}%"),
    )
    columns = st.columns(4)
    for column, (label, value) in zip(columns, cards):
        with column:
            st.markdown(metric_card(label, value), unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(ear_timeline(events), use_container_width=True)
    with right:
        st.plotly_chart(ear_histogram(events), use_container_width=True)
    st.plotly_chart(drowsiness_timeline(events), use_container_width=True)
    st.markdown(section_header("Logged Events"), unsafe_allow_html=True)
    st.dataframe(events, use_container_width=True, hide_index=True)


def build_report_data(session_id: str, events: pd.DataFrame) -> dict[str, Any]:
    """Build report metadata and event records from a saved session CSV."""
    timestamps = pd.to_datetime(
        events.get("timestamp"), errors="coerce", utc=True
    ).dropna()
    start_time = (
        timestamps.min().isoformat() if not timestamps.empty else "Unavailable"
    )
    end_time = (
        timestamps.max().isoformat() if not timestamps.empty else "Unavailable"
    )
    duration = (
        (timestamps.max() - timestamps.min()).total_seconds()
        if len(timestamps) > 1
        else 0
    )
    for column in ("session_start_time", "session_end_time"):
        if column in events and events[column].notna().any():
            value = str(events[column].dropna().iloc[0])
            if column == "session_start_time":
                start_time = value
            else:
                end_time = value
    if (
        "duration_seconds" in events
        and events["duration_seconds"].notna().any()
    ):
        duration = float(
            pd.to_numeric(events["duration_seconds"], errors="coerce")
            .dropna()
            .iloc[0]
        )
    return {
        "session_id": session_id,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration,
        "events": events,
    }


def render_reports() -> None:
    """Render report generation, summary preview, and prior downloads."""
    st.markdown(
        section_header("Reports", "Generate and download a session PDF."),
        unsafe_allow_html=True,
    )
    sessions = list_sessions()
    if not sessions:
        st.info("No saved sessions are available for reporting.")
    else:
        selected_session = st.selectbox("Report session", sessions)
        try:
            events = load_session_csv(selected_session)
            summary = summarize_events(events)
            if summary["average_ear"] is not None:
                report_preview = (
                    f"**Session:** {selected_session}  \n"
                    f"**Alerts:** {summary['alerts']}  \n"
                    f"**Average EAR:** {summary['average_ear']:.3f}"
                )
            else:
                report_preview = (
                    f"**Session:** {selected_session}  \n"
                    f"**Alerts:** {summary['alerts']}  \n"
                    "**Average EAR:** N/A"
                )
            st.markdown(report_preview)
            if st.button("Generate PDF Report", type="primary"):
                report_data = build_report_data(selected_session, events)
                report_path = ReportGenerator(
                    report_data, REPORT_DIR
                ).generate_pdf()
                st.session_state.generated_report = str(report_path)
                st.success("PDF report generated.")
            report_path_value = st.session_state.get("generated_report")
            if report_path_value:
                report_path = Path(report_path_value)
                if report_path.is_file():
                    st.download_button(
                        "Download PDF Report",
                        data=report_path.read_bytes(),
                        file_name=report_path.name,
                        mime="application/pdf",
                    )
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            LOGGER.exception("Unable to generate session report")
            st.error(f"Unable to generate the report: {error}")
    st.markdown(
        section_header("Previously Generated Reports"), unsafe_allow_html=True
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_files = sorted(
        REPORT_DIR.glob("*.pdf"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not report_files:
        st.caption("No PDF reports have been generated.")
    for report_path in report_files:
        try:
            st.download_button(
                f"Download {report_path.name}",
                data=report_path.read_bytes(),
                file_name=report_path.name,
                mime="application/pdf",
                key=f"report_{report_path.name}",
            )
        except OSError as error:
            LOGGER.exception("Unable to read report file %s", report_path)
            st.error(f"Unable to read {report_path.name}: {error}")


def render_settings(detector: DrowsinessDetector) -> None:
    """Render thresholds, EAR explanation, model status, and data reset."""
    st.markdown(
        section_header(
            "Settings", "Tune detector behavior and manage local data."
        ),
        unsafe_allow_html=True,
    )
    detector.ear_threshold = st.slider(
        "EAR Threshold",
        min_value=0.15,
        max_value=0.35,
        value=0.25,
        step=0.01,
        key="ear_threshold",
    )
    detector.consecutive_frames = st.slider(
        "Consecutive Frames",
        min_value=10,
        max_value=40,
        value=20,
        step=1,
        key="consecutive_frames",
    )
    st.toggle("Alarm Sound", value=True, key="alarm_enabled")
    st.markdown("### Eye Aspect Ratio (EAR)")
    st.write(
        "EAR estimates eye openness from six points around each eye. Lower "
        "values indicate a more closed eye."
    )
    st.latex(
        r"\mathrm{EAR} = \frac{\lVert p_2-p_6\rVert + "
        r"\lVert p_3-p_5\rVert}{2\lVert p_1-p_4\rVert}"
    )
    model_exists = MODEL_PATH.is_file()
    st.markdown(
        status_banner(
            "Model available" if model_exists else "Model missing",
            str(MODEL_PATH.relative_to(ROOT)),
        ),
        unsafe_allow_html=True,
    )
    st.warning(
        "Reset permanently removes saved session CSV files and generated "
        "PDF reports."
    )
    detection_active = bool(st.session_state.get("detection_active", False))
    if detection_active:
        st.info("Stop the active detection session before resetting its data.")
    if st.button(
        "Reset Session Data",
        type="secondary",
        disabled=detection_active,
    ):
        if st.session_state.get("confirm_reset"):
            reset_failed = False
            for path in (*LOG_DIR.glob("*.csv"), *REPORT_DIR.glob("*.pdf")):
                try:
                    path.unlink()
                except OSError as error:
                    reset_failed = True
                    LOGGER.exception("Unable to remove %s", path)
                    st.error(f"Unable to remove {path.name}: {error}")
            st.session_state.confirm_reset = False
            if not reset_failed:
                st.success(
                    "Session CSV data and generated reports were reset."
                )
        else:
            st.session_state.confirm_reset = True
            st.warning("Click Reset Session Data again to confirm.")


def main() -> None:
    """Route navigation to the selected application page."""
    detector = get_detector()
    page = st.sidebar.radio(
        "Navigation",
        (
            "🏠 Home / Dashboard",
            "🎥 Live Detection",
            "📊 Analytics Dashboard",
            "📄 Reports",
            "⚙️ Settings",
        ),
    )
    if page.startswith("🏠"):
        render_home(detector)
    elif page.startswith("🎥"):
        render_live_detection(detector)
    elif page.startswith("📊"):
        render_analytics()
    elif page.startswith("📄"):
        render_reports()
    else:
        render_settings(detector)


if __name__ == "__main__":
    main()
