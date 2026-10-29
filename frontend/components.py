"""HTML component builders for the Streamlit interface."""

from html import escape


def metric_card(label: str, value: str, detail: str = "") -> str:
    """Return a styled metric card as safe HTML."""
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{escape(value)}</div>'
        f'<div class="metric-detail">{escape(detail)}</div>'
        "</div>"
    )


def status_banner(status: str, message: str = "") -> str:
    """Return a status banner with a semantic color class."""
    normalized = status.lower()
    color_class = (
        "status-danger"
        if "drowsy" in normalized
        else (
            "status-warning"
            if "blink" in normalized
            else (
                "status-success"
                if normalized in ("active", "running")
                else "status-neutral"
            )
        )
    )
    return (
        f'<div class="status-banner {color_class}">'
        f"<strong>{escape(status)}</strong>"
        f"<span>{escape(message)}</span>"
        "</div>"
    )


def info_card(title: str, body: str) -> str:
    """Return a styled information card as safe HTML."""
    return (
        '<div class="info-card">'
        f"<h3>{escape(title)}</h3>"
        f"<p>{escape(body)}</p>"
        "</div>"
    )


def section_header(title: str, subtitle: str = "") -> str:
    """Return a section heading and optional subtitle as safe HTML."""
    return (
        '<div class="section-header">'
        f"<h2>{escape(title)}</h2>"
        f"<p>{escape(subtitle)}</p>"
        "</div>"
    )
