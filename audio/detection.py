import asyncio
from typing import Optional
import numpy as np

from agent.realtime.event_bus import EventBus
from agent.state.base import VoiceAssistantEvent
from audio.capture import AudioCapture
from shared.logging_mixin import LoggingMixin


class AudioDetectionService(LoggingMixin):
    """Service for Audio-Level Detection and Speech Detection with EventBus integration"""

    def __init__(
        self,
        audio_capture: AudioCapture,
        threshold: float = 40.0,
        check_interval: float = 0.1,
    ):
        self.audio_capture = audio_capture
        self.threshold = threshold
        self.check_interval = check_interval
        self.event_bus = EventBus()

        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False

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
                    await self._trigger_speech_detected()
                    break  # Stop monitoring after detection

                await asyncio.sleep(self.check_interval)

        except asyncio.CancelledError:  # NOSONAR
            self.logger.debug("Audio monitoring cancelled")
        except Exception as e:
            self.logger.exception("Audio monitoring failed: %s", e)
            await self._trigger_error(e)

    async def _trigger_speech_detected(self) -> None:
        """Trigger speech detected via EventBus"""
        try:
            self.event_bus.publish_sync(VoiceAssistantEvent.USER_STARTED_SPEAKING)
        except Exception as e:
            self.logger.exception("Error triggering speech detected: %s", e)

    async def _trigger_error(self, error: Exception) -> None:
        """Trigger error via EventBus"""
        try:
            self.event_bus.publish_sync(VoiceAssistantEvent.ERROR_OCCURRED)
        except Exception as e:
            self.logger.exception("Error triggering error callback: %s", e)

    def update_threshold(self, new_threshold: float) -> None:
        """Update detection threshold"""
        self.threshold = new_threshold
        self.logger.info("Updated threshold to: %.1f", new_threshold)