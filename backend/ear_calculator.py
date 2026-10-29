"""Eye Aspect Ratio calculations and eye landmark extraction."""

from collections.abc import Sequence

import numpy as np
from scipy.spatial import distance

LEFT_EYE_INDICES = tuple(range(42, 48))
RIGHT_EYE_INDICES = tuple(range(36, 42))


def calculate_ear(eye_points: np.ndarray | Sequence[Sequence[float]]) -> float:
    """Calculate the Eye Aspect Ratio for six ordered eye landmarks."""
    points = np.asarray(eye_points, dtype=float)
    if points.shape != (6, 2):
        raise ValueError(
            "eye_points must contain exactly six two-dimensional points."
        )
    vertical_one = distance.euclidean(points[1], points[5])
    vertical_two = distance.euclidean(points[2], points[4])
    horizontal = distance.euclidean(points[0], points[3])
    if horizontal == 0:
        raise ValueError(
            "The horizontal eye landmark distance cannot be zero."
        )
    return float((vertical_one + vertical_two) / (2.0 * horizontal))


def get_eye_landmarks(
    landmarks: np.ndarray | Sequence[Sequence[float]],
    eye_indices: Sequence[int],
) -> np.ndarray:
    """Return the six two-dimensional landmarks for one eye."""
    points = np.asarray(landmarks)
    indices = np.asarray(eye_indices, dtype=int)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(
            "landmarks must be a two-dimensional array with x and y values."
        )
    if indices.shape != (6,):
        raise ValueError("eye_indices must contain exactly six indices.")
    if np.any(indices < 0) or np.any(indices >= len(points)):
        raise IndexError(
            "eye_indices contain an index outside the landmark array."
        )
    return points[indices, :2].astype(float, copy=True)
