from agent.state.base import AssistantState, StateType, VoiceAssistantEvent
from agent.state.context import VoiceAssistantContext


class ListeningState(AssistantState):
    """State when listening for user input after they started speaking"""

    def __init__(self):
        super().__init__(StateType.LISTENING)

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info("Entering Listening state - user is speaking")
        self._ensure_realtime_audio_channel_connected(context)

        self.logger.debug("Initiating realtime session for user conversation")
        success = await context.start_realtime_session()
        if not success:
            self.logger.error("Failed to initiate realtime session in listening state")

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        # Nothing to do here - realtime session cleanup handled by context/state machine
        pass

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        match event:
            case VoiceAssistantEvent.USER_SPEECH_ENDED:
                self.logger.info("User finished speaking")
                return await self._transition_to_responding(context)
            case VoiceAssistantEvent.ERROR_OCCURRED:
                await self._transition_to_error(context)
            case _:
                self.logger.debug("Ignoring event %s in Listening state", event.value)

            
    def _ensure_realtime_audio_channel_connected(self, context: VoiceAssistantContext) -> None:
        """Ensure realtime audio channel is connected"""
        if not context.audio_capture.is_active:
            context.audio_capture.start_stream()
            self.logger.info("Microphone stream reactivated")
        else:
            self.logger.debug("Microphone stream already active")
            
        context.resume_realtime_audio()