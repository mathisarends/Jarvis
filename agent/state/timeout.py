from agent.state.base import AssistantState, StateType, VoiceAssistantEvent
from agent.state.context import VoiceAssistantContext


class TimeoutState(AssistantState):
    """State after wake word - waiting for user to start speaking with timeout"""

    def __init__(self):
        super().__init__(StateType.TIMEOUT)

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info(
            "Entering TimeoutState - user has %s seconds to start speaking",
            context.timeout_service.timeout_seconds,
        )
        await context.ensure_realtime_audio_channel_connected()

        # Start both timeout service and audio detection
        await self._start_timeout_service(context)
        await self._start_audio_detection(context)

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        await self._stop_timeout_service(context)

        # Only stop audio detection if we're transitioning to idle
        # If transitioning to listening, audio detection should continue
        from agent.state.base import StateType

        if context.state.state_type == StateType.IDLE:
            await context.event_bus.publish_sync(VoiceAssistantEvent.IDLE_TRANSITION)

            self.logger.info("Closing realtime connection due to timeout")
            await context.realtime_api.close_connection()
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
        """Start the timeout service"""
        self.logger.debug("Starting timeout service")
        # No need to set callbacks anymore - service uses EventBus directly
        await context.timeout_service.start_timeout()

    async def _stop_timeout_service(self, context: VoiceAssistantContext) -> None:
        """Stop the timeout service"""
        self.logger.debug("Stopping timeout service")
        await context.timeout_service.stop_timeout()

    async def _start_audio_detection(self, context: VoiceAssistantContext) -> None:
        """Start audio detection using AudioDetectionService"""
        self.logger.debug("Starting audio detection for speech detection")
        # No need to set callbacks anymore - service uses EventBus directly
        await context.audio_detection_service.start_monitoring()

    async def _stop_audio_detection(self, context: VoiceAssistantContext) -> None:
        """Stop audio detection"""
        self.logger.debug("Stopping audio detection")
        await context.audio_detection_service.stop_monitoring()
