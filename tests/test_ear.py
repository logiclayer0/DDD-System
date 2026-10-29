"""Tests for Eye Aspect Ratio calculations."""

import numpy as np
import pytest

from backend.ear_calculator import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    calculate_ear,
    get_eye_landmarks,
)


def test_calculate_ear_for_open_eye() -> None:
    """Calculate a positive EAR for a symmetric open eye."""
    points = np.array(
        [[0, 0], [1, -1], [3, -1], [4, 0], [3, 1], [1, 1]],
        dtype=float,
    )
    assert calculate_ear(points) == pytest.approx(0.5)


def test_calculate_ear_rejects_zero_horizontal_distance() -> None:
    """Reject degenerate points that would divide by zero."""
    points = np.zeros((6, 2))
    with pytest.raises(ValueError, match="horizontal"):
        calculate_ear(points)


def test_get_eye_landmarks_uses_eye_index_sets() -> None:
    """Extract six eye points from a 68-point landmark matrix."""
    landmarks = np.arange(136).reshape(68, 2)
    assert np.array_equal(
        get_eye_landmarks(landmarks, LEFT_EYE_INDICES), landmarks[42:48]
    )
    assert np.array_equal(
        get_eye_landmarks(landmarks, RIGHT_EYE_INDICES), landmarks[36:42]
    )


def test_get_eye_landmarks_rejects_out_of_range_index() -> None:
    """Reject eye indexes that exceed the landmark array."""
    with pytest.raises(IndexError):
        get_eye_landmarks(np.zeros((10, 2)), RIGHT_EYE_INDICES)
