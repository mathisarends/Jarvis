from agent.state.base import AssistantState, StateType, VoiceAssistantEvent
from agent.state.context import VoiceAssistantContext
from audio.sound_player import SoundFile


class ErrorState(AssistantState):
    """Error state - catch-all for handling errors"""

    def __init__(self):
        super().__init__(StateType.ERROR)

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.error("Entering Error state - handling error condition")
        context.sound_player.play_sound_file(SoundFile.ERROR)

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        # No cleanup needed when exiting error state
        pass

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        match event:
            case VoiceAssistantEvent.ASSISTANT_RESPONSE_COMPLETED:
                # After error message is delivered, return to idle
                self.logger.info("Error response completed, returning to idle")
                await self._transition_to_idle(context)
            case VoiceAssistantEvent.IDLE_TRANSITION:
                # Direct transition to idle
                self.logger.info("Idle transition in Error state")
                await self._transition_to_idle(context)
            case _:
                self.logger.debug("Ignoring event %s in Error state", event.value)
