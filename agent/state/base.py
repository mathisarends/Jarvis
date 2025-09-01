"""
State machine for voice assistant - consolidated states (match/case)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from audio import SoundFile
from shared.logging_mixin import LoggingMixin

if TYPE_CHECKING:
    from audio import SoundFilePlayer


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


class StateType(Enum):
    """Enum for different state types"""

    IDLE = "idle"
    LISTENING = "listening"
    RESPONDING = "responding"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


class VoiceAssistantContext:
    """Context object that holds state and dependencies"""

    def __init__(self, sound_player: SoundFilePlayer):
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
    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        """Handle an event in this state"""
        ...

    async def _transition_to(
        self, new_state: "AssistantState", context: VoiceAssistantContext
    ) -> None:
        """Transition to a new state"""
        self.logger.info(
            "Transitioning from %s to %s",
            self.__class__.__name__,
            new_state.__class__.__name__,
        )
        context.state = new_state
        await context.state.on_enter(context)


class IdleState(AssistantState):
    """Initial state - waiting for wake word"""

    def __init__(self):
        super().__init__(StateType.IDLE)

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info("Entering Idle state - waiting for wake word")
        context.end_session()

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        match event:
            case VoiceAssistantEvent.WAKE_WORD_DETECTED:
                context.start_session()
                await self._transition_to(ListeningState(), context)
            case _:
                self.logger.debug("Ignoring event %s in Idle state", event.value)


class ListeningState(AssistantState):
    """State when listening for user input after wake word"""

    def __init__(self):
        super().__init__(StateType.LISTENING)

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info("Entering Listening state - ready for user input")
        # Wake-word sound cue
        context.sound_player.play_sound_file(SoundFile.WAKE_WORD)

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        match event:
            case VoiceAssistantEvent.USER_INPUT_RECEIVED:
                await self._transition_to(RespondingState(), context)
            case VoiceAssistantEvent.SESSION_TIMEOUT:
                self.logger.info("Session timeout in Listening state")
                await self._transition_to(IdleState(), context)
            case VoiceAssistantEvent.ERROR_OCCURRED:
                await self._transition_to(ErrorState(), context)
            case _:
                self.logger.debug("Ignoring event %s in Listening state", event.value)


class RespondingState(AssistantState):
    """State when generating and delivering response to user"""

    def __init__(self):
        super().__init__(StateType.RESPONDING)

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info(
            "Entering Responding state - generating and delivering response"
        )

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        match event:
            case VoiceAssistantEvent.SPEECH_DONE | VoiceAssistantEvent.SPEECH_INTERRUPTED:
                await self._transition_to(ListeningState(), context)
            case VoiceAssistantEvent.SESSION_TIMEOUT:
                self.logger.info("Session timeout in Responding state")
                await self._transition_to(IdleState(), context)
            case VoiceAssistantEvent.ERROR_OCCURRED:
                await self._transition_to(ErrorState(), context)
            case _:
                self.logger.debug("Ignoring event %s in Responding state", event.value)


class ErrorState(AssistantState):
    """Error state - catch-all for handling errors"""

    def __init__(self):
        super().__init__(StateType.ERROR)

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.error("Entering Error state - handling error condition")
        context.sound_player.play_sound_file(SoundFile.ERROR)

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        match event:
            case VoiceAssistantEvent.SPEECH_DONE | VoiceAssistantEvent.SESSION_TIMEOUT:
                await self._transition_to(IdleState(), context)
            case _:
                self.logger.debug("Ignoring event %s in Error state", event.value)
