from agent.state.base import State, VoiceAssistantContext, VoiceAssistantEvent
from agent.state.listening import Listening


class IdleState(State):
    """Initial state - waiting for wake word"""
    
    def on_enter(self, context: VoiceAssistantContext) -> None:
        """Enter idle state"""
        self.logger.info("Entering Idle state - waiting for wake word")
        context.end_session()
    
    def handle(self, event: VoiceAssistantEvent, context: VoiceAssistantContext) -> None:
        """Handle events in idle state"""
        if event == VoiceAssistantEvent.WAKE_WORD_DETECTED:
            context.start_session()
            self._transition_to(Listening(), context)
        else:
            self.logger.debug("Ignoring event %s in Idle state", event.value)
