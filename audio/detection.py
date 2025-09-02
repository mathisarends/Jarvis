import asyncio
from typing import Callable, Optional
import numpy as np
from audio.capture import AudioCapture
from shared.logging_mixin import LoggingMixin


class AudioDetectionService(LoggingMixin):
    """Service for Audio-Level Detection and Speech Detection"""

    def __init__(
        self,
        audio_capture: AudioCapture,
        threshold: float = 40.0,
        check_interval: float = 0.1,
    ):
        self.audio_capture = audio_capture
        self.threshold = threshold
        self.check_interval = check_interval

        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False

        # Callbacks
        self._on_speech_detected: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

    def set_speech_callback(self, callback: Callable) -> None:
        """Set callback that is called when speech is detected"""
        self._on_speech_detected = callback

    def set_error_callback(self, callback: Callable) -> None:
        """Set callback that is called on errors"""
        self._on_error = callback

    async def start_monitoring(self) -> None:
        """Start audio monitoring"""
        if self._is_monitoring:
            self.logger.warning("Audio monitoring already active")
            return

        self.logger.info(
            "Starting audio monitoring with threshold: %.1f", self.threshold
        )
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self) -> None:
        """Stop audio monitoring"""
        if not self._is_monitoring:
            return

        self.logger.info("Stopping audio monitoring")
        self._is_monitoring = False

        if not self._monitoring_task or self._monitoring_task.done():
            return

        self._monitoring_task.cancel()
        try:
            await self._monitoring_task
        except asyncio.CancelledError:  # NOSONAR
            # disable linting for state transition
            pass
        finally:
            self._monitoring_task = None

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        try:
            while self._is_monitoring:
                # Read audio chunk
                audio_data = self.audio_capture.read_chunk()
                if audio_data is None:
                    await asyncio.sleep(self.check_interval)
                    continue

                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                audio_level = np.sqrt(np.mean(audio_array**2))
                self.logger.debug(
                    f"Audio level: {audio_level:.1f} (threshold: {self.threshold})"
                )

                # Check if audio level exceeds threshold
                if audio_level > self.threshold:
                    self.logger.info("Speech detected (level: %.1f)", audio_level)
                    await self._trigger_speech_detected(audio_level)
                    break  # Stop monitoring after detection

                await asyncio.sleep(self.check_interval)

        except asyncio.CancelledError:  # NOSONAR
            self.logger.debug("Audio monitoring cancelled")
        except Exception as e:
            self.logger.exception("Audio monitoring failed: %s", e)
            await self._trigger_error(e)

    async def _trigger_speech_detected(self, audio_level: float) -> None:
        """Trigger speech detected callback"""
        if not self._on_speech_detected:
            return
        try:
            if asyncio.iscoroutinefunction(self._on_speech_detected):
                await self._on_speech_detected(audio_level)
            else:
                self._on_speech_detected(audio_level)
        except Exception as e:
            self.logger.exception("Error in speech detected callback: %s", e)

    async def _trigger_error(self, error: Exception) -> None:
        """Trigger error callback"""
        if not self._on_error:
            return
        try:
            if asyncio.iscoroutinefunction(self._on_error):
                await self._on_error(error)
            else:
                self._on_error(error)
        except Exception as e:
            self.logger.exception("Error in error callback: %s", e)

    def update_threshold(self, new_threshold: float) -> None:
        """Update detection threshold"""
        self.threshold = new_threshold
        self.logger.info("Updated threshold to: %.1f", new_threshold)

    @property
    def is_monitoring(self) -> bool:
        """Check if currently monitoring"""
        return self._is_monitoring
