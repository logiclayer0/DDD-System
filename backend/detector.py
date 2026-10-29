"""Thread-safe facial landmark and drowsiness detection engine."""

import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.ear_calculator import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    calculate_ear,
    get_eye_landmarks,
)

LOGGER = logging.getLogger(__name__)
try:
    import dlib
except ImportError:
    dlib = None


class DrowsinessDetector:
    """Detect drowsiness from consecutive low-EAR frames."""

    def __init__(
        self,
        ear_threshold: float = 0.25,
        consecutive_frames: int = 20,
        model_path: str | Path | None = None,
    ) -> None:
        """Initialize thresholds, model paths, and synchronized state."""
        self.ear_threshold = float(ear_threshold)
        self.consecutive_frames = int(consecutive_frames)
        self.model_path = (
            Path(model_path)
            if model_path
            else (
                Path(__file__).resolve().parents[1]
                / "models"
                / "shape_predictor_68_face_landmarks.dat"
            )
        )
        self._face_detector: Any | None = None
        self._predictor: Any | None = None
        self._lock = threading.RLock()
        self._eyes_closed_frames = 0

    @property
    def models_loaded(self) -> bool:
        """Return whether both dlib models are initialized."""
        return self._face_detector is not None and self._predictor is not None

    def load_models(self) -> bool:
        """Load the face detector and configured landmark predictor."""
        with self._lock:
            if dlib is None:
                LOGGER.error(
                    "dlib is not installed; facial landmark models "
                    "cannot load."
                )
                return False
            if not self.model_path.is_file():
                LOGGER.warning(
                    "Landmark model was not found at %s", self.model_path
                )
                return False
            try:
                self._face_detector = dlib.get_frontal_face_detector()
                self._predictor = dlib.shape_predictor(str(self.model_path))
            except (RuntimeError, OSError) as error:
                self._face_detector = None
                self._predictor = None
                LOGGER.exception(
                    "Unable to load facial landmark models: %s", error
                )
                return False
            LOGGER.info(
                "Facial landmark models loaded from %s", self.model_path
            )
            return True

    def detect_face(
        self, frame: np.ndarray
    ) -> tuple[int, int, int, int] | None:
        """Return the largest face's bounding box."""
        if self._face_detector is None:
            return None
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(gray_frame, 0)
        if not faces:
            return None
        face = max(
            faces, key=lambda rectangle: rectangle.width() * rectangle.height()
        )
        return face.left(), face.top(), face.right(), face.bottom()

    def get_landmarks(
        self,
        frame: np.ndarray,
        face: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Return 68 facial landmark coordinates for a face box."""
        if self._predictor is None:
            raise RuntimeError(
                "The facial landmark model has not been loaded."
            )
        left, top, right, bottom = face
        if dlib is None:
            raise RuntimeError("dlib is not installed.")
        rectangle = dlib.rectangle(left, top, right, bottom)
        shape = self._predictor(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), rectangle
        )
        return np.asarray(
            [
                (shape.part(index).x, shape.part(index).y)
                for index in range(68)
            ],
            dtype=int,
        )

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        """Analyze a frame and return EAR, state, count, and overlay data."""
        with self._lock:
            if frame is None or frame.ndim not in (2, 3):
                raise ValueError("frame must be a valid image array.")
            if not self.models_loaded:
                self._eyes_closed_frames = 0
                return {
                    "ear": None,
                    "is_drowsy": False,
                    "eyes_closed_frames": 0,
                    "status": "Model Not Loaded",
                    "face": None,
                    "landmarks": None,
                }
            face = self.detect_face(frame)
            if face is None:
                self._eyes_closed_frames = 0
                return {
                    "ear": None,
                    "is_drowsy": False,
                    "eyes_closed_frames": 0,
                    "status": "No Face Detected",
                    "face": None,
                    "landmarks": None,
                }
            landmarks = self.get_landmarks(frame, face)
            left_eye = get_eye_landmarks(landmarks, LEFT_EYE_INDICES)
            right_eye = get_eye_landmarks(landmarks, RIGHT_EYE_INDICES)
            ear_value = (
                calculate_ear(left_eye) + calculate_ear(right_eye)
            ) / 2.0
            if ear_value < self.ear_threshold:
                self._eyes_closed_frames += 1
            else:
                self._eyes_closed_frames = 0
            is_drowsy = self._eyes_closed_frames >= self.consecutive_frames
            status = (
                "DROWSY!"
                if is_drowsy
                else ("Blinking" if self._eyes_closed_frames else "Active")
            )
            return {
                "ear": ear_value,
                "is_drowsy": is_drowsy,
                "eyes_closed_frames": self._eyes_closed_frames,
                "status": status,
                "face": face,
                "landmarks": landmarks,
            }

    def reset_state(self) -> None:
        """Reset the consecutive closed-eye frame counter."""
        with self._lock:
            self._eyes_closed_frames = 0
