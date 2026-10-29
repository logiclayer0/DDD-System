"""Tests for detector frame-state behavior."""

import numpy as np
import pytest

from backend.detector import DrowsinessDetector


def test_no_face_resets_closed_eye_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a clear no-face status and reset the frame count."""
    detector = DrowsinessDetector()
    detector._face_detector = object()
    detector._predictor = object()
    detector._eyes_closed_frames = 7
    monkeypatch.setattr(detector, "detect_face", lambda frame: None)
    result = detector.process_frame(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result["status"] == "No Face Detected"
    assert result["ear"] is None
    assert result["eyes_closed_frames"] == 0


def test_missing_models_return_explicit_status() -> None:
    """Report an unavailable model instead of a missing face."""
    detector = DrowsinessDetector()
    result = detector.process_frame(np.zeros((20, 20, 3), dtype=np.uint8))
    assert result["status"] == "Model Not Loaded"


def test_reset_state_clears_closed_eye_counter() -> None:
    """Reset the detector's accumulated closed-eye count."""
    detector = DrowsinessDetector()
    detector._eyes_closed_frames = 4
    detector.reset_state()
    assert (
        detector.process_frame(np.zeros((20, 20, 3), dtype=np.uint8))[
            "eyes_closed_frames"
        ]
        == 0
    )


def test_sustained_eye_closure_triggers_drowsy_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mark drowsiness after the configured number of low-EAR frames."""
    detector = DrowsinessDetector(consecutive_frames=2)
    detector._face_detector = object()
    detector._predictor = object()
    eye_points = np.array(
        [[0, 0], [1, -0.2], [2, -0.2], [4, 0], [2, 0.2], [1, 0.2]],
        dtype=float,
    )
    landmarks = np.zeros((68, 2), dtype=float)
    landmarks[36:42] = eye_points
    landmarks[42:48] = eye_points
    monkeypatch.setattr(detector, "detect_face", lambda frame: (0, 0, 10, 10))
    monkeypatch.setattr(
        detector,
        "get_landmarks",
        lambda frame, face: landmarks,
    )

    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    first_result = detector.process_frame(frame)
    second_result = detector.process_frame(frame)

    assert first_result["status"] == "Blinking"
    assert second_result["status"] == "DROWSY!"
    assert second_result["is_drowsy"] is True
