"""
VoiceAssistantContext - Updated to include wake word listener
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from audio.capture import AudioCapture
from audio.detection import AudioDetectionService
from agent.state.timeout_service import TimeoutService
from audio.sound_player import SoundPlayer
from shared.logging_mixin import LoggingMixin

if TYPE_CHECKING:
    from audio import So
    from audio.wake_word_listener import WakeWordListener
    from agent.state.base import AssistantState
    from agent.state.base import VoiceAssistantEvent


class VoiceAssistantContext(LoggingMixin):
    """Context object that holds state and dependencies"""

    def __init__(
        self,
        sound_player: SoundPlayer,
        wake_word_listener: WakeWordListener,
        audio_capture: AudioCapture,
        audio_detection_service: AudioDetectionService,
        timeout_service: TimeoutService,
    ):
        # Needed for initialization - prevent circular deps
        from agent.state.idle import IdleState

        self.state: AssistantState = IdleState()
        self.session_active = False
        self.sound_player = sound_player
        self.wake_word_listener = wake_word_listener
        self.audio_capture = audio_capture
        self.audio_detection_service = audio_detection_service
        self.timeout_service = timeout_service

    async def handle_event(self, event: VoiceAssistantEvent) -> None:
        """Delegate event to current state"""
        await self.state.handle(event, self)

    def start_session(self) -> None:
        """Start a new session"""
        self.session_active = True

    def end_session(self) -> None:
        """End the current session"""
        self.session_active = False

    def is_idle(self) -> bool:
        """Check if the current state is idle"""
        from agent.state.base import StateType

        return self.state.state_type == StateType.IDLE
