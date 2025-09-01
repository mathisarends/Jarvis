"""
VoiceAssistantContext - Updated to include wake word listener
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shared.logging_mixin import LoggingMixin

if TYPE_CHECKING:
    from audio import SoundFilePlayer
    from audio.wake_word_listener import WakeWordListener
    from agent.state.base import AssistantState
    from agent.state.base import VoiceAssistantEvent


class VoiceAssistantContext(LoggingMixin):
    """Context object that holds state and dependencies"""

    def __init__(
        self, sound_player: SoundFilePlayer, wake_word_listener: WakeWordListener
    ):
        # Import hier statt oben - vermeidet circular import
        from agent.state.idle import IdleState

        self.state: AssistantState = IdleState()
        self.session_active = False
        self.sound_player = sound_player
        self.wake_word_listener = wake_word_listener

    async def handle_event(self, event: VoiceAssistantEvent) -> None:
        """Delegate event to current state"""
        await self.state.handle(event, self)

    def start_session(self) -> None:
        """Start a new session"""
        self.session_active = True

    def end_session(self) -> None:
        """End the current session"""
        self.session_active = False
