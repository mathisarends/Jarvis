from agent.state.base import AssistantState, StateType, VoiceAssistantEvent
from agent.state.context import VoiceAssistantContext


class ListeningState(AssistantState):
    """State when listening for user input after they started speaking"""

    def __init__(self):
        super().__init__(StateType.LISTENING)

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info("Entering Listening state - user is speaking")

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        # Nothing to clean up in listening state
        pass

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        match event:
            case VoiceAssistantEvent.USER_SPEECH_ENDED:
                self.logger.info("User finished speaking")
                await self._transition_to_responding(context)
            case VoiceAssistantEvent.IDLE_TRANSITION:
                self.logger.info("Idle transition in Listening state")
                await self._transition_to_idle(context)
            case VoiceAssistantEvent.ERROR_OCCURRED:
                await self._transition_to_error(context)
            case _:
                self.logger.debug("Ignoring event %s in Listening state", event.value)
