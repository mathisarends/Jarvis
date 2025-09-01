from agent.state.base import State, VoiceAssistantContext, VoiceAssistantEvent
from agent.state.idle import IdleState
from agent.state.responding import RespondingState
from agent.state.error import ErrorState

class ListeningState(State):
    """State when listening for user input after wake word"""
    
    def on_enter(self, context: VoiceAssistantContext) -> None:
        """Enter listening state"""
        self.logger.info("Entering Listening state - ready for user input")
    
    def handle(self, event: VoiceAssistantEvent, context: VoiceAssistantContext) -> None:
        """Handle events in listening state"""
        if event == VoiceAssistantEvent.USER_INPUT_RECEIVED:
            self._transition_to(RespondingState(), context)
        elif event == VoiceAssistantEvent.SESSION_TIMEOUT:
            self.logger.info("Session timeout in Listening state")
            self._transition_to(IdleState(), context)
        elif event == VoiceAssistantEvent.ERROR_OCCURRED:
            self._transition_to(ErrorState(), context)
        else:
            self.logger.debug("Ignoring event %s in Listening state", event.value)
