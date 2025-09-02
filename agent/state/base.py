"""
State machine for voice assistant - base classes and enums
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from agent.state.context import VoiceAssistantContext
from shared.logging_mixin import LoggingMixin


class VoiceAssistantEvent(Enum):
    """Events that can trigger state transitions"""

    WAKE_WORD_DETECTED = "wake_word_detected"
    USER_STARTED_SPEAKING = "user_started_speaking"
    USER_SPEECH_ENDED = "user_speech_ended"

    ASSISTANT_STARTED_RESPONDING = "assistant_started_responding"
    ASSISTANT_RESPONSE_COMPLETED = "assistant_response_completed"
    ASSISTANT_SPEECH_INTERRUPTED = "assistant_speech_interrupted"

    ASSISTANT_STARTED_TOOL_CALL = "assistant_started_tool_call"
    ASSISTANT_RECEIVED_TOOL_CALL_RESULT = "ASSISTANT_RECEIVED_TOOL_CALL_RESULT"

    TIMEOUT_OCCURRED = "timeout_occurred"
    IDLE_TRANSITION = "idle_transition"
    ERROR_OCCURRED = "error_occurred"

    def __str__(self) -> str:
        return self.value


class StateType(Enum):
    """Enum for different state types"""

    IDLE = "idle"
    TIMEOUT = "timeout"
    LISTENING = "listening"
    RESPONDING = "responding"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


class AssistantState(ABC, LoggingMixin):
    """Base class for all states"""

    def __init__(self, state_type: StateType):
        super().__init__()
        self._state_type = state_type

    @property
    def state_type(self) -> StateType:
        """Read-only property that returns the state type"""
        return self._state_type

    @abstractmethod
    async def on_enter(self, context: VoiceAssistantContext) -> None:
        """Called when entering this state"""
        ...

    @abstractmethod
    async def on_exit(self, context: VoiceAssistantContext) -> None:
        """Called when exiting this state"""
        ...

    @abstractmethod
    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        """Handle an event in this state"""
        ...

    async def _transition_to_idle(self, context: VoiceAssistantContext) -> None:
        """Transition to IdleState"""
        from agent.state.idle import IdleState

        await self._transition_to(IdleState(), context)

    async def _transition_to_timeout(self, context: VoiceAssistantContext) -> None:
        """Transition to TimeoutState"""
        from agent.state.timeout import TimeoutState

        await self._transition_to(TimeoutState(), context)

    async def _transition_to_listening(self, context: VoiceAssistantContext) -> None:
        """Transition to ListeningState"""
        from agent.state.listening import ListeningState

        await self._transition_to(ListeningState(), context)

    async def _transition_to_responding(self, context: VoiceAssistantContext) -> None:
        """Transition to RespondingState"""
        from agent.state.responding import RespondingState

        await self._transition_to(RespondingState(), context)

    async def _transition_to_error(self, context: VoiceAssistantContext) -> None:
        """Transition to ErrorState"""
        from agent.state.error import ErrorState

        await self._transition_to(ErrorState(), context)

    async def _transition_to(
        self, new_state: AssistantState, context: VoiceAssistantContext
    ) -> None:
        """Transition to a new state"""
        self.logger.info(
            "Transitioning from %s to %s",
            self.__class__.__name__,
            new_state.__class__.__name__,
        )
        context.state = new_state

        await self.on_exit(context)
        await context.state.on_enter(context)
