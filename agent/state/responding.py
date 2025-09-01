from agent.state.base import State, VoiceAssistantContext, VoiceAssistantEvent
from agent.state.idle import IdleState
from agent.state.listening import ListeningState
from agent.state.error import ErrorState

class RespondingState(State):
    """State when generating and delivering response to user"""
    
    def on_enter(self, context: VoiceAssistantContext) -> None:
        """Enter responding state"""
        self.logger.info("Entering Responding state - generating and delivering response")
    
    def handle(self, event: VoiceAssistantEvent, context: VoiceAssistantContext) -> None:
        """Handle events in responding state"""
        if event == VoiceAssistantEvent.SPEECH_DONE:
            # After response, go back to listening for more input in the same session
            self._transition_to(ListeningState(), context)
        elif event == VoiceAssistantEvent.SPEECH_INTERRUPTED:
            # If interrupted, go back to listening
            self._transition_to(ListeningState(), context)
        elif event == VoiceAssistantEvent.SESSION_TIMEOUT:
            self.logger.info("Session timeout in Responding state")
            self._transition_to(IdleState(), context)
        elif event == VoiceAssistantEvent.ERROR_OCCURRED:
            self._transition_to(ErrorState(), context)
        else:
            self.logger.debug("Ignoring event %s in Responding state", event.value)
