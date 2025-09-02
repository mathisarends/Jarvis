from agent.state.base import AssistantState, StateType, VoiceAssistantEvent
from agent.state.context import VoiceAssistantContext
from audio.sound_player import SoundFile


class TimeoutState(AssistantState):
    """State after wake word - waiting for user to start speaking with timeout"""

    def __init__(self):
        super().__init__(StateType.TIMEOUT)

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info(
            "Entering TimeoutState - user has %s seconds to start speaking",
            context.timeout_service.timeout_seconds,
        )

        context.sound_player.play_sound_file(SoundFile.WAKE_WORD)

        context.audio_capture.start_stream()

        # Start both timeout service and audio detection
        await self._start_timeout_service(context)
        await self._start_audio_detection(context)

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        await self._stop_timeout_service(context)

        # Only stop audio detection if we're transitioning to idle
        # If transitioning to listening, audio detection should continue
        if context.is_idle():
            await self._stop_audio_detection(context)

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
                    context.timeout_service.timeout_seconds,
                )
                await self._transition_to_idle(context)
            case VoiceAssistantEvent.IDLE_TRANSITION:
                self.logger.info("Idle transition in TimeoutState")
                await self._transition_to_idle(context)
            case VoiceAssistantEvent.ERROR_OCCURRED:
                await self._transition_to_error(context)
            case _:
                self.logger.debug("Ignoring event %s in TimeoutState", event.value)

    async def _start_timeout_service(self, context: VoiceAssistantContext) -> None:
        """Start the timeout service with callbacks"""
        self.logger.debug("Starting timeout service")

        async def on_timeout_callback() -> None:
            self.logger.info("Timeout occurred - sending TIMEOUT_OCCURRED event")
            await context.handle_event(VoiceAssistantEvent.TIMEOUT_OCCURRED)

        # Set callbacks on the timeout service
        context.timeout_service.set_timeout_callback(on_timeout_callback)

        # Start timeout
        await context.timeout_service.start_timeout()

    async def _stop_timeout_service(self, context: VoiceAssistantContext) -> None:
        """Stop the timeout service"""
        self.logger.debug("Stopping timeout service")
        await context.timeout_service.stop_timeout()

    async def _start_audio_detection(self, context: VoiceAssistantContext) -> None:
        """Start audio detection using AudioDetectionService"""
        self.logger.debug("Starting audio detection for speech detection")

        async def on_speech_detected_callback(audio_level: float) -> None:
            self.logger.info(
                "User speech detected by audio detection service (level: %.1f)",
                audio_level,
            )
            await context.handle_event(VoiceAssistantEvent.USER_STARTED_SPEAKING)

        async def on_error_callback(error: Exception) -> None:
            self.logger.error("Audio detection error: %s", error)
            await context.handle_event(VoiceAssistantEvent.ERROR_OCCURRED)

        # Set callbacks on the detection service
        context.audio_detection_service.set_speech_callback(on_speech_detected_callback)
        context.audio_detection_service.set_error_callback(on_error_callback)

        # Start monitoring
        await context.audio_detection_service.start_monitoring()

    async def _stop_audio_detection(self, context: VoiceAssistantContext) -> None:
        """Stop audio detection"""
        self.logger.debug("Stopping audio detection")
        await context.audio_detection_service.stop_monitoring()
