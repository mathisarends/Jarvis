"""
State machine for voice assistant
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from shared.logging_mixin import LoggingMixin


class VoiceAssistantEvent(Enum):
    """Events that can trigger state transitions"""
    WAKE_WORD_DETECTED = "wake_word_detected"
    USER_INPUT_RECEIVED = "user_input_received"
    RESPONSE_GENERATED = "response_generated"
    SESSION_TIMEOUT = "session_timeout"
    ERROR_OCCURRED = "error_occurred"
    SPEECH_DONE = "speech_done"
    SPEECH_INTERRUPTED = "speech_interrupted"

    def __str__(self) -> str:
        return self.value


class VoiceAssistantContext:
    """Context object that holds state and dependencies"""
    def __init__(self):
        self.state: State = self._get_initial_state()
        self.session_active = False
        self.state.on_enter(self)

    def handle_event(self, event: VoiceAssistantEvent) -> None:
        """Handle an event by delegating to current state"""
        self.state.handle(event, self)

    def start_session(self) -> None:
        """Start a new session"""
        self.session_active = True

    def end_session(self) -> None:
        """End the current session"""
        self.session_active = False
        
    def _get_initial_state(self) -> State:
        """Lazy import to avoid circular imports"""
        from agent.state.idle import Idle
        return Idle()



class State(ABC, LoggingMixin):
    """Base class for all states"""
    
    @abstractmethod
    def on_enter(self, context: VoiceAssistantContext) -> None:
        """Called when entering this state"""
        pass
    
    @abstractmethod
    def handle(self, event: VoiceAssistantEvent, context: VoiceAssistantContext) -> None:
        """Handle an event in this state"""
        pass

    def _transition_to(self, new_state: State, context: VoiceAssistantContext) -> None:
        """Helper method to transition to a new state"""
        self.logger.info("Transitioning from %s to %s", 
                        self.__class__.__name__, 
                        new_state.__class__.__name__)
        context.state = new_state
        context.state.on_enter(context)