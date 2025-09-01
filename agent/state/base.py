"""
State machine for voice assistant - base classes and enums
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from shared.logging_mixin import LoggingMixin

if TYPE_CHECKING:
    from audio import SoundFilePlayer


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
    WAITING_FOR_USER_INPUT = "waiting_for_user_input"
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

    async def _transition_to(
        self, new_state: AssistantState, context: VoiceAssistantContext
    ) -> None:
        """Transition to a new state"""
        self.logger.info(
            "Transitioning from %s to %s",
            self.__class__.__name__,
            new_state.__class__.__name__,
        )
        
        await self.on_exit(context)
        
        context.state = new_state
        await context.state.on_enter(context)


class VoiceAssistantContext:
    """Context object that holds state and dependencies"""

    def __init__(self, sound_player: SoundFilePlayer):
        # Import hier statt oben - vermeidet circular import
        from agent.state.idle import IdleState
        
        self.state: AssistantState = IdleState()
        self.session_active = False
        self.sound_player = sound_player

    async def handle_event(self, event: VoiceAssistantEvent) -> None:
        """Delegate event to current state"""
        await self.state.handle(event, self)

    def start_session(self) -> None:
        """Start a new session"""
        self.session_active = True

    def end_session(self) -> None:
        """End the current session"""
        self.session_active = False