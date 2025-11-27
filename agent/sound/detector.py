import asyncio
from typing import Optional
import numpy as np

from agent.realtime.event_bus import EventBus
from agent.state.base import VoiceAssistantEvent
from agent.mic import MicrophoneCapture
from shared.logging_mixin import LoggingMixin


class AudioDetectionService(LoggingMixin):
    def __init__(
        self,
        audio_capture: MicrophoneCapture,
        event_bus: EventBus,
        threshold: float = 40.0,
        check_interval: float = 0.1,
    ):
        self.audio_capture = audio_capture
        self.event_bus = event_bus
        self.threshold = threshold
        self.check_interval = check_interval

        self._monitoring_task: asyncio.Task | None = None
        self._is_monitoring = False

    async def start_monitoring(self) -> None:
        if self._is_monitoring:
            self.logger.warning("Audio monitoring already active")
            return

        self.logger.info(
            "Starting audio monitoring with threshold: %.1f", self.threshold
        )
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self) -> None:
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
        """Polling-based monitoring loop for audio level detection"""
        try:
            while self._is_monitoring:
                # Read current audio chunk
                audio_data = self.audio_capture.read_chunk()

                if audio_data and self._process_audio_chunk(audio_data):
                    break

                # Sleep before next poll
                await asyncio.sleep(self.check_interval)

        except asyncio.CancelledError:  # NOSONAR
            self.logger.debug("Audio monitoring cancelled")
        except Exception as e:
            self.logger.exception("Audio monitoring failed: %s", e)
            self._trigger_error(e)

    def _process_audio_chunk(self, audio_data: bytes) -> bool:
        """
        Process a single audio chunk and check for speech.
        """
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Calculate RMS (Root Mean Square) audio level
            audio_level = np.sqrt(np.mean(audio_array**2))
            self.logger.debug(
                f"Audio level: {audio_level:.1f} (threshold: {self.threshold})"
            )

            # Check if audio level exceeds threshold
            if audio_level > self.threshold:
                self.logger.info("Speech detected (level: %.1f)", audio_level)
                self._trigger_speech_detected()
                return True

            return False

        except Exception as e:
            self.logger.error("Error processing audio chunk: %s", e)
            return False

    def _trigger_speech_detected(self) -> None:
        self.event_bus.publish_sync(VoiceAssistantEvent.USER_STARTED_SPEAKING)

    def _trigger_error(self, error: Exception) -> None:
        self.event_bus.publish_sync(VoiceAssistantEvent.ERROR_OCCURRED)
