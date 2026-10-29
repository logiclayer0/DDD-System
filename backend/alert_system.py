"""Cross-platform alarm playback with a repeat cooldown."""

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class AlertSystem:
    """Play an alarm without repeating it during the three-second cooldown."""

    def __init__(self, sound_file: str | Path = "assets/alarm.wav") -> None:
        """Initialize the sound path and synchronized playback state."""
        self.sound_file = Path(sound_file)
        if not self.sound_file.is_absolute():
            self.sound_file = (
                Path(__file__).resolve().parents[1] / self.sound_file
            )
        self._lock = threading.Lock()
        self._last_started = 0.0
        self._playing = False
        self._process: subprocess.Popen[bytes] | None = None
        self._stop_event = threading.Event()

    def _play_in_thread(self) -> None:
        try:
            if sys.platform == "win32":
                import winsound

                winsound.PlaySound(
                    str(self.sound_file),
                    winsound.SND_FILENAME | winsound.SND_ASYNC,
                )
                while not self._stop_event.wait(0.1):
                    if not self._playing:
                        break
            elif sys.platform == "darwin":
                self._process = subprocess.Popen(
                    ["afplay", str(self.sound_file)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._process.wait()
            else:
                self._process = subprocess.Popen(
                    ["aplay", "-q", str(self.sound_file)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._process.wait()
        except (OSError, RuntimeError) as error:
            LOGGER.warning("Unable to play the alarm sound: %s", error)
        finally:
            with self._lock:
                self._playing = False
                self._process = None

    def play_alarm(self) -> bool:
        """Start alarm playback when no playback or cooldown is active."""
        with self._lock:
            now = time.monotonic()
            if self._playing or now - self._last_started < 3.0:
                return False
            if not self.sound_file.is_file():
                LOGGER.warning(
                    "Alarm sound was not found at %s", self.sound_file
                )
                return False
            self._last_started = now
            self._playing = True
            self._stop_event.clear()
        thread = threading.Thread(target=self._play_in_thread, daemon=True)
        thread.start()
        return True

    def stop_alarm(self) -> None:
        """Stop alarm playback when platform interruption is supported."""
        self._stop_event.set()
        with self._lock:
            self._playing = False
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        if sys.platform == "win32":
            try:
                import winsound

                winsound.PlaySound(None, winsound.SND_PURGE)
            except (RuntimeError, ImportError) as error:
                LOGGER.warning(
                    "Unable to stop Windows alarm playback: %s", error
                )

    def is_playing(self) -> bool:
        """Return whether alarm playback is active."""
        with self._lock:
            return self._playing
