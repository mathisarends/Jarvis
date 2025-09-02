import asyncio
from typing import Optional
import numpy as np
from agent.state.base import AssistantState, StateType, VoiceAssistantEvent
from agent.state.context import VoiceAssistantContext
from audio.file_player import SoundFile


# Das hier mit den audio thresholds funktioniert hier noch nicht leider:
class TimeoutState(AssistantState):
    """State after wake word - waiting for user to start speaking with timeout"""

    TIMEOUT_SECONDS = 10.0
    AUDIO_THRESHOLD = 40  # Adjust based on your microphone sensitivity
    AUDIO_CHECK_INTERVAL = 0.1  # Check audio every 100ms

    def __init__(self):
        super().__init__(StateType.TIMEOUT)
        self._timeout_task: Optional[asyncio.Task] = None
        self._audio_monitoring_task: Optional[asyncio.Task] = None

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info(
            "Entering TimeoutState - user has %s seconds to start speaking",
            self.TIMEOUT_SECONDS,
        )

        context.sound_player.play_sound_file(SoundFile.WAKE_WORD)

        context.audio_capture.start_stream()

        # Start both timeout and audio monitoring
        await self._start_timeout(context)
        await self._start_audio_monitoring(context)

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        await self._cancel_timeout()
        await self._cancel_audio_monitoring()

        # Stop audio stream
        context.audio_capture.stop_stream()

    async def _start_timeout(self, context: VoiceAssistantContext) -> None:
        """Start the timeout task for this state"""
        await self._cancel_timeout()  # Cancel any existing timeout

        self.logger.debug("Starting timeout: %s seconds", self.TIMEOUT_SECONDS)
        self._timeout_task = asyncio.create_task(self._timeout_handler(context))

    async def _start_audio_monitoring(self, context: VoiceAssistantContext) -> None:
        """Start audio monitoring task"""
        await self._cancel_audio_monitoring()

        self.logger.debug("Starting audio monitoring")
        self._audio_monitoring_task = asyncio.create_task(
            self._audio_monitoring_loop(context)
        )

    async def _timeout_handler(self, context: VoiceAssistantContext) -> None:
        """Handle timeout by sending TIMEOUT_OCCURRED event"""
        try:
            await asyncio.sleep(self.TIMEOUT_SECONDS)
            self.logger.info("Timeout occurred after %s seconds", self.TIMEOUT_SECONDS)
            await context.handle_event(VoiceAssistantEvent.TIMEOUT_OCCURRED)
        except asyncio.CancelledError:  # NOSONAR
            # Intentional cleanup - CancelledError not re-raised to avoid aborting state transition
            self.logger.debug("Timeout cancelled")

    async def _audio_monitoring_loop(self, context: VoiceAssistantContext) -> None:
        """Monitor audio for speech detection"""
        try:
            while True:
                # Read audio chunk
                audio_data = context.audio_capture.read_chunk()
                if audio_data is None:
                    await asyncio.sleep(self.AUDIO_CHECK_INTERVAL)
                    continue

                # Convert to numpy array for analysis
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                # Calculate audio level (RMS)
                audio_level = np.sqrt(np.mean(audio_array**2))
                print("audio_level", audio_level)

                # Check if audio level exceeds threshold
                if audio_level > self.AUDIO_THRESHOLD:
                    self.logger.info("User speech detected (level: %.1f)", audio_level)
                    await context.handle_event(
                        VoiceAssistantEvent.USER_STARTED_SPEAKING
                    )
                    break

                await asyncio.sleep(self.AUDIO_CHECK_INTERVAL)

        except asyncio.CancelledError:  # NOSONAR
            # Intentional cleanup - CancelledError not re-raised to avoid aborting state transition
            self.logger.debug("Audio monitoring cancelled")
        except Exception:
            self.logger.exception("Audio monitoring failed")
            await context.handle_event(VoiceAssistantEvent.ERROR_OCCURRED)

    async def _cancel_timeout(self) -> None:
        """Cancel any running timeout"""
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:  # NOSONAR
                # Intentional cleanup - CancelledError not re-raised to avoid aborting state transition
                pass
            finally:
                self._timeout_task = None

    async def _cancel_audio_monitoring(self) -> None:
        """Cancel audio monitoring task"""
        if self._audio_monitoring_task and not self._audio_monitoring_task.done():
            self._audio_monitoring_task.cancel()
            try:
                await self._audio_monitoring_task
            except asyncio.CancelledError:  # NOSONAR
                # Intentional cleanup - CancelledError not re-raised to avoid aborting state transition
                pass
            finally:
                self._audio_monitoring_task = None

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        match event:
            case VoiceAssistantEvent.USER_STARTED_SPEAKING:
                self.logger.info("User started speaking - transitioning to listening")
                await self._transition_to_listening(context)
            case VoiceAssistantEvent.TIMEOUT_OCCURRED:
                self.logger.info(
                    "Timeout occurred - user did not start speaking within %s seconds",
                    self.TIMEOUT_SECONDS,
                )
                await self._transition_to_idle(context)
            case VoiceAssistantEvent.IDLE_TRANSITION:
                self.logger.info("Idle transition in TimeoutState")
                await self._transition_to_idle(context)
            case VoiceAssistantEvent.ERROR_OCCURRED:
                await self._transition_to_error(context)
            case _:
                self.logger.debug("Ignoring event %s in TimeoutState", event.value)
