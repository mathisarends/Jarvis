"""
State machine for voice assistant - consolidated states
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
        self.state: AssistantState = IdleState()
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


class AssistantState(ABC, LoggingMixin):
    """Base class for all states"""

    @abstractmethod
    def on_enter(self, context: VoiceAssistantContext) -> None:
        """Called when entering this state"""
        pass

    @abstractmethod
    def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        """Handle an event in this state"""
        pass

    def _transition_to(
        self, new_state: AssistantState, context: VoiceAssistantContext
    ) -> None:
        """Helper method to transition to a new state"""
        self.logger.info(
            "Transitioning from %s to %s",
            self.__class__.__name__,
            new_state.__class__.__name__,
        )
        context.state = new_state
        context.state.on_enter(context)


class IdleState(AssistantState):
    """Initial state - waiting for wake word"""

    def on_enter(self, context: VoiceAssistantContext) -> None:
        """Enter idle state"""
        self.logger.info("Entering Idle state - waiting for wake word")
        context.end_session()

    def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        """Handle events in idle state"""
        if event == VoiceAssistantEvent.WAKE_WORD_DETECTED:
            context.start_session()
            self._transition_to(ListeningState(), context)
        else:
            self.logger.debug("Ignoring event %s in Idle state", event.value)


class ListeningState(AssistantState):
    """State when listening for user input after wake word"""

    def on_enter(self, context: VoiceAssistantContext) -> None:
        """Enter listening state"""
        self.logger.info("Entering Listening state - ready for user input")

    def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
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


class RespondingState(AssistantState):
    """State when generating and delivering response to user"""

    def on_enter(self, context: VoiceAssistantContext) -> None:
        """Enter responding state"""
        self.logger.info(
            "Entering Responding state - generating and delivering response"
        )

    def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        """Handle events in responding state"""
        if event == VoiceAssistantEvent.SPEECH_DONE:
            self._transition_to(ListeningState(), context)
        elif event == VoiceAssistantEvent.SPEECH_INTERRUPTED:
            self._transition_to(ListeningState(), context)
        elif event == VoiceAssistantEvent.SESSION_TIMEOUT:
            self.logger.info("Session timeout in Responding state")
            self._transition_to(IdleState(), context)
        elif event == VoiceAssistantEvent.ERROR_OCCURRED:
            self._transition_to(ErrorState(), context)
        else:
            self.logger.debug("Ignoring event %s in Responding state", event.value)


class ErrorState(AssistantState):
    """Error state - catch-all for handling errors"""

    def on_enter(self, context: VoiceAssistantContext) -> None:
        """Enter error state"""
        self.logger.error("Entering Error state - handling error condition")

    def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
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
