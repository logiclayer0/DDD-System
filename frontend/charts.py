"""Interactive Plotly visualizations for session analytics."""

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

NAVY = "#0A192F"
CYAN = "#64FFDA"
RED = "#D64545"


def _frame_events(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    if "event_type" in frame.columns:
        frame = frame[frame["event_type"].eq("frame")]
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], errors="coerce", utc=True
    )
    frame["ear_value"] = pd.to_numeric(frame["ear_value"], errors="coerce")
    return frame.dropna(subset=["timestamp", "ear_value"])


def ear_timeline(events: pd.DataFrame) -> Figure:
    """Build the interactive EAR-over-time line chart."""
    frame = _frame_events(events)
    figure = px.line(
        frame,
        x="timestamp",
        y="ear_value",
        title="Eye Aspect Ratio Over Time",
        labels={"timestamp": "Time", "ear_value": "Eye Aspect Ratio"},
        color_discrete_sequence=[CYAN],
        markers=True,
    )
    figure.update_layout(
        template="plotly_white", font_color=NAVY, hovermode="x unified"
    )
    return figure


def ear_histogram(events: pd.DataFrame) -> Figure:
    """Build the EAR distribution histogram."""
    frame = _frame_events(events)
    figure = px.histogram(
        frame,
        x="ear_value",
        nbins=25,
        title="EAR Distribution",
        labels={"ear_value": "Eye Aspect Ratio", "count": "Frame Count"},
        color_discrete_sequence=[NAVY],
    )
    figure.update_layout(template="plotly_white", font_color=NAVY)
    return figure


def drowsiness_timeline(events: pd.DataFrame) -> Figure:
    """Build a timeline highlighting drowsy frames and alert events."""
    frame = events.copy()
    if "timestamp" not in frame.columns:
        frame["timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
    if "ear_value" not in frame.columns:
        frame["ear_value"] = pd.Series(dtype=float)
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], errors="coerce", utc=True
    )
    frame["ear_value"] = pd.to_numeric(frame["ear_value"], errors="coerce")
    drowsy_mask = (
        frame["is_drowsy"].astype(str).str.lower().isin(("true", "1", "yes"))
        if "is_drowsy" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    alert_mask = (
        frame["event_type"].eq("alert")
        if "event_type" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    points = frame[drowsy_mask | alert_mask].copy()
    points["event_label"] = "Drowsy frame"
    points.loc[alert_mask.loc[points.index], "event_label"] = "Alert"
    points.loc[points["event_label"].eq("Alert"), "ear_value"] = 0.0
    figure = px.scatter(
        points,
        x="timestamp",
        y="ear_value",
        color="event_label",
        symbol="event_label",
        title="Drowsiness Events",
        labels={
            "timestamp": "Time",
            "ear_value": "Eye Aspect Ratio",
            "event_label": "Event",
        },
        color_discrete_map={"Drowsy frame": "#F08C46", "Alert": RED},
        hover_data=["status"] if "status" in points.columns else None,
    )
    figure.update_traces(marker={"size": 10})
    figure.update_layout(
        template="plotly_white",
        font_color=NAVY,
        hovermode="closest",
    )
    return figure
