from agent.state.base import State, VoiceAssistantContext, VoiceAssistantEvent
from agent.state.idle import IdleState


class ErrorState(State):
    """Error state - catch-all for handling errors"""
    
    def on_enter(self, context: VoiceAssistantContext) -> None:
        """Enter error state"""
        self.logger.error("Entering Error state - handling error condition")
    
    def handle(self, event: VoiceAssistantEvent, context: VoiceAssistantContext) -> None:
        """Handle events in error state"""
        if event == VoiceAssistantEvent.SPEECH_DONE:
            # After error message is delivered, return to idle
            self._transition_to(IdleState(), context)
        elif event == VoiceAssistantEvent.SESSION_TIMEOUT:
            # Timeout during error handling
            self.logger.info("Session timeout in Error state")
            self._transition_to(IdleState(), context)
        else:
            self.logger.debug("Ignoring event %s in Error state", event.value)