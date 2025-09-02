from agent.state.base import AssistantState, StateType, VoiceAssistantEvent
from agent.state.context import VoiceAssistantContext


class ListeningState(AssistantState):
    """State when listening for user input after they started speaking"""

    def __init__(self):
        super().__init__(StateType.LISTENING)

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info("Entering Listening state - user is speaking")

        # Reactivate microphone stream if it was deactivated (e.g., during assistant response)
        if not context.audio_capture.is_active:
            context.audio_capture.start_stream()
            self.logger.info(
                "Microphone stream reactivated - ready to capture user speech"
            )
        else:
            self.logger.debug("Microphone stream already active")

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        # Log microphone stream status before exiting
        if context.audio_capture.is_active:
            self.logger.debug(
                "Microphone stream still active when exiting Listening state"
            )
        else:
            self.logger.debug("Microphone stream was already stopped")

        # Note: We don't stop the audio stream here as RespondingState might need it
        self.logger.info("Exiting Listening state")

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        match event:
            case VoiceAssistantEvent.USER_SPEECH_ENDED:
                self.logger.info("User finished speaking")
                return await self._transition_to_responding(context)
            case VoiceAssistantEvent.IDLE_TRANSITION:
                self.logger.info("Idle transition in Listening state")
                await self._transition_to_idle(context)
            case VoiceAssistantEvent.ERROR_OCCURRED:
                await self._transition_to_error(context)
            case _:
                self.logger.debug("Ignoring event %s in Listening state", event.value)
